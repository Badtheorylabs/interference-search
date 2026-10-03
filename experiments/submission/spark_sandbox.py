"""Unprivileged Linux Landlock + seccomp sandbox for generated MBPP programs.

Spark disallows bubblewrap user/network namespaces. Landlock is enabled and
requires no host privilege change. The generated program cannot read the home
directory, write host files, open sockets, spawn processes, or signal other jobs.
Setup failures are infrastructure failures and stop the experiment.
"""
import json
import subprocess
import tempfile
from pathlib import Path

CHILD = r'''
import ast, contextlib, ctypes, io, json, os, resource, signal, sys

payload = json.load(sys.stdin)
source, tests = payload["source"], payload["tests"]
resource.setrlimit(resource.RLIMIT_CPU, (3,3))
resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024,512*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536,65536))
resource.setrlimit(resource.RLIMIT_NOFILE, (32,32))
resource.setrlimit(resource.RLIMIT_CORE, (0,0))
libc = ctypes.CDLL(None,use_errno=True)
seccomp = ctypes.CDLL("/lib/aarch64-linux-gnu/libseccomp.so.2")

class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs",ctypes.c_uint64)]
class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access",ctypes.c_uint64),("parent_fd",ctypes.c_int32)]
def checked(value):
    if value < 0:
        raise OSError(ctypes.get_errno(),"sandbox setup failed")
    return value
try:
    abi = checked(libc.syscall(444,0,0,1))
    if abi < 3:
        raise RuntimeError("Landlock ABI 3+ required")
    rule = Ruleset((1<<15)-1)
    fd = checked(libc.syscall(444,ctypes.byref(rule),ctypes.sizeof(rule),0))
    read_exec = (1<<0)|(1<<2)|(1<<3)
    for path in ("/usr","/lib","/lib64"):
        if not os.path.exists(path):
            continue
        parent = os.open(path,os.O_PATH|os.O_CLOEXEC)
        item = PathRule(read_exec,parent)
        checked(libc.syscall(445,fd,1,ctypes.byref(item),0))
        os.close(parent)
    for path in ("/dev/null","/dev/urandom"):
        parent = os.open(path,os.O_PATH|os.O_CLOEXEC)
        item = PathRule((1<<2)|((1<<1) if path=="/dev/null" else 0),parent)
        checked(libc.syscall(445,fd,1,ctypes.byref(item),0))
        os.close(parent)
    checked(libc.prctl(38,1,0,0,0))  # PR_SET_NO_NEW_PRIVS
    checked(libc.syscall(446,fd,0))
    os.close(fd)
    seccomp.seccomp_init.argtypes=[ctypes.c_uint32]
    seccomp.seccomp_init.restype=ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    seccomp.seccomp_load.argtypes=[ctypes.c_void_p]
    seccomp.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=seccomp.seccomp_init(0x7fff0000)
    if not ctx:
        raise RuntimeError("seccomp init failed")
    for name in ("socket","socketpair","connect","accept","accept4","bind","listen",
                 "sendto","sendmsg","recvfrom","recvmsg","execve","execveat","clone",
                 "clone3","fork","vfork","kill","tkill","tgkill","ptrace","bpf",
                 "mount","umount2","unshare","setns","process_vm_readv","process_vm_writev",
                 "pidfd_open","pidfd_getfd","pidfd_send_signal","io_uring_setup","keyctl"):
        number=seccomp.seccomp_syscall_resolve_name(name.encode())
        if number>=0:
            checked(seccomp.seccomp_rule_add(ctx,0x50000|1,number,0))
    checked(seccomp.seccomp_load(ctx))
    seccomp.seccomp_release(ctx)
except BaseException as error:
    print(json.dumps({"infrastructure_error":str(error)}))
    sys.exit(2)

def alarm(*args):
    raise TimeoutError("per-test timeout")
signal.signal(signal.SIGALRM,alarm)
g={}
results=[]
try:
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        signal.alarm(2)
        for statement in payload.get("imports",[]):
            exec(compile(statement,"<import>","exec"),g)
        exec(compile(source,"<candidate>","exec"),g)
        signal.alarm(0)
except BaseException as error:
    signal.alarm(0)
    results=[["load_error",type(error).__name__+": "+str(error)[:160]]]*len(tests)
else:
    for test in tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                signal.alarm(2)
                exec(compile(test,"<test>","exec"),g)
                signal.alarm(0)
            results.append(["pass",""])
        except BaseException as error:
            signal.alarm(0)
            results.append(["fail",type(error).__name__+": "+str(error)[:160]])
print(json.dumps({"results":results}))
'''


def execute(source, tests, imports=None, timeout=8):
    if not tests or len(source) > 50000:
        raise ValueError("invalid candidate")
    payload = json.dumps({"source":source,"tests":tests,"imports":imports or []})
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.run(["/usr/bin/python3","-I","-S","-c",CHILD],input=payload,
                                     stdout=stdout,stderr=stderr,text=True,timeout=timeout)
            stdout.seek(0)
            output=stdout.read(65536).decode(errors="replace")
            stderr.seek(0)
            error=stderr.read(2048).decode(errors="replace")
        response=json.loads(output.splitlines()[-1])
        if "infrastructure_error" in response:
            raise RuntimeError(response["infrastructure_error"])
        return response["results"]
    except subprocess.TimeoutExpired:
        return [["timeout","wall limit"] for _ in tests]
    except (ValueError,IndexError):
        return [["candidate_crash","runner exited"] for _ in tests]


def preflight():
    assert execute("def f(): return 3",["assert f() == 3"]) == [["pass",""]]
    restrictions = [
        ("open('/home/badtheory/.ssh/authorized_keys').read()",["assert True"]),
        ("open('/tmp/is-sandbox-escape','w').write('x')",["assert True"]),
        ("import socket; socket.socket()",["assert True"]),
        ("import os; os.fork()",["assert True"]),
    ]
    for source, tests in restrictions:
        result=execute(source,tests)
        assert result[0][0] == "load_error",(source,result)
    return {"landlock":True,"seccomp":True,"filesystem_network_process_probes":4}


if __name__ == "__main__":
    print(json.dumps(preflight()))

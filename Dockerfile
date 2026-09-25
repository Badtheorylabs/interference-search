# CPU image for the Countdown experiments (not yet built on a clean machine; please open an issue if it fails). The language-model experiments need MLX on Apple silicon
# and are not included.
#   docker build -t interference-search .
#   docker run --rm interference-search
FROM python:3.11-slim
WORKDIR /repo
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN pip install --no-cache-dir -e ".[test]"
CMD ["bash", "scripts/reproduce_countdown.sh"]

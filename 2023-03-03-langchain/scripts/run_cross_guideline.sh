mkdir -p metrics/cross/stab2018
python -m scripts.evaluate outputs/langchain-test-stab2018.jsonl outputs/langchain-test-stab2018.jsonl --output-path metrics/cross/stab2018/langchain-test-stab2018.jsonl --normalize
python -m scripts.evaluate outputs/langchain-test-stab2018.jsonl outputs/langchain-test-levy2018.jsonl --output-path metrics/cross/stab2018/langchain-test-levy2018.jsonl --normalize
python -m scripts.evaluate outputs/langchain-test-stab2018.jsonl outputs/langchain-test-shnarch2018.jsonl --output-path metrics/cross/stab2018/langchain-test-shnarch2018.jsonl --normalize
mkdir -p metrics/cross/levy2018
python -m scripts.evaluate outputs/langchain-test-levy2018.jsonl outputs/langchain-test-stab2018.jsonl --output-path metrics/cross/levy2018/langchain-test-stab2018.jsonl --normalize
python -m scripts.evaluate outputs/langchain-test-levy2018.jsonl outputs/langchain-test-levy2018.jsonl --output-path metrics/cross/levy2018/langchain-test-levy2018.jsonl --normalize
python -m scripts.evaluate outputs/langchain-test-levy2018.jsonl outputs/langchain-test-shnarch2018.jsonl --output-path metrics/cross/levy2018/langchain-test-shnarch2018.jsonl --normalize
mkdir -p metrics/cross/shnarch2018
python -m scripts.evaluate outputs/langchain-test-shnarch2018.jsonl outputs/langchain-test-stab2018.jsonl --output-path metrics/cross/shnarch2018/langchain-test-stab2018.jsonl --normalize
python -m scripts.evaluate outputs/langchain-test-shnarch2018.jsonl outputs/langchain-test-levy2018.jsonl --output-path metrics/cross/shnarch2018/langchain-test-levy2018.jsonl --normalize
python -m scripts.evaluate outputs/langchain-test-shnarch2018.jsonl outputs/langchain-test-shnarch2018.jsonl --output-path metrics/cross/shnarch2018/langchain-test-shnarch2018.jsonl --normalize
pip install --break-system-packages --only-binary=:all: 'pyarrow>=14'
python3 -c "import pyarrow; print(f'pyarrow version: {pyarrow.__version__}')"

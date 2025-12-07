generate python code from root of repository. they end up in lib

    python -m grpc.tools.protoc -I protos --python_betterproto2_out=lib protos/*.proto

https://betterproto.github.io/python-betterproto2/getting-started/

    from lib.nf.telemetry import TelemetryBatchUpdate
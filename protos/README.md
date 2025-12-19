generate python code from root of repository. they end up in generated

    python -m grpc.tools.protoc -I protos --python_betterproto2_out=generated protos/*.proto

https://betterproto.github.io/python-betterproto2/getting-started/

    from generated.nf.telemetry import TelemetryBatchUpdate
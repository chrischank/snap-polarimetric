FROM up42/up42-snap-py38:latest

ARG .
ARG manifest
LABEL "up42_manifest"=$manifest

# Copy the SNAP toolbox configuration.
COPY $BUILD_DIR/gpt.vmoptions /usr/local/snap/bin/
COPY $BUILD_DIR/snap.properties /usr/local/snap/etc/

WORKDIR /block

COPY $BUILD_DIR/requirements.txt /block
RUN pip3 install -r requirements.txt
COPY $BUILD_DIR/src/ /block/src/

ENV LD_LIBRARY_PATH="."

CMD ["python3", "/block/src/run.py"]

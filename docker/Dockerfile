FROM darribas/gds_py:6.0

LABEL maintainer="Martin Fleischmann <martin@martinfleischmann.net>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mamba install --yes --quiet cython \
    && mamba remove pygeos --yes --quiet \
    && pip install git+git://github.com/martinfleis/pygeos.git@b9cdd1d81791bf704e60f11230c7988de8d56434 --no-build-isolation

ENV GDS_ENV_VERSION "6.0dev_pygeos"
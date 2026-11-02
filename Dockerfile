FROM gurobi/optimizer:11.0.3

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --no-cache-dir \ 
    matplotlib \
    pandas \
    numpy \
    CoolProp \
    scipy \
    openpyxl \
    pyomo \
    gurobipy==11.0.3

ENV GUROBI_HOME=/opt/gurobi/linux
ENV PATH="${GUROBI_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
ENV GRB_LICENSE_FILE=/software/groups/gurobi_opt/gurobi.lic

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /
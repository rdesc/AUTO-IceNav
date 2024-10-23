# Installing HSL solvers for use with IPOPT (optional)
This is a guide to install the HSL solvers which can be used with IPOPT + CasADi.
### Easy install installation if running Ubuntu and x86-64

Install dependencies:
````
apt install libblas-dev liblapack-dev libmetis-dev
````

Copy the file `libhsl.so` to where the CasADi package is installed and contains `casadi.py`.
For example:
```
cp libhsl.so ~/miniconda3/envs/py39/lib/python3.9/site-packages/casadi
```

### Longer installation process
Instructions are a little hard to follow and may be outdated but the following links 
should be sufficient for figuring out how to obtain the `libhsl.so` or `libhsl.dll` file.
Once obtained, copy the file to the same directory as `casadi.py`.

Links:
- https://github.com/casadi/casadi/wiki/Obtaining-HSL
- https://github.com/coin-or-tools/ThirdParty-HSL
- https://licences.stfc.ac.uk/product/coin-hsl
- https://coin-or.github.io/Ipopt/INSTALL.html#DOWNLOAD_HSL

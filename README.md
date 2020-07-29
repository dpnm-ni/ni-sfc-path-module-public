# ni-sfc-sub-module

## Main Responsibilities
Random or RL-based SFC path selection module.
- Provide APIs to create random SFCs
- Provide APIs to create SFCs by using Q-learning

## Requirements
```
Python 3.5.2+
```

Please install pip3 and requirements by using the command as below. 
```
sudo apt-get update
sudo apt-get insatll python3-pip
pip3 install -r requirements.txt
```

## Usage
If you installed all requirements in advance, you can run this module by using the command as below.
```
python3 -m swagger_server
```
and open your browser to here:

```
http://localhost:8888/ui/
```

Your Swagger definition lives here:

```
http://localhost:8888/swagger.json
```

This module will be integrated with ni-ai-module. Further, this module design comes from ni-nfvo-module and ni-ai-module. So, if you want to know detailed design, you can see the modules. 

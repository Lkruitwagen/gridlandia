import yaml, os

generation_technologies = yaml.load(open(os.path.join(os.path.split(__file__)[0],'generation_technologies.yml'),'r'), Loader=yaml.SafeLoader)
transmission_technologies = yaml.load(open(os.path.join(os.path.split(__file__)[0],'transmission_technologies.yml'),'r'),Loader=yaml.SafeLoader)

__all__ = ["generation_technologies","transmission_technologies"]
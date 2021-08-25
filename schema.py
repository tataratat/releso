from SbSOvRL.base_parser import BaseParser
import json


with open("schema.json", "w") as f:
    f.write(BaseParser.schema_json(indent=2))
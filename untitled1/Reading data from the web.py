import pandas as pd
import numpy as np
import requests

url = "https://www.wikipedia.org/"
r = requests.get(url)
text = r.text

print (text)

# TWITTER API PROJECT
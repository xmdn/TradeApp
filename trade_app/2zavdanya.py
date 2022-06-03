import numpy
import libpysal
import spreg
from esda.moran import Moran
from libpysal.weights import Queen, KNN
import seaborn
import pandas
import geopandas
import pyproj
import matplotlib.pyplot as plt

libpysal.examples.explain("baltim")


#читання балтімор даних
db = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
ds_name = "baltim.dbf"

#читання залежностей змінних
y_name = "PRICE"
y = numpy.array(db.by_col(y_name)).T
y = y[:, numpy.newaxis]

#читання екзогенних змінних
x_names = ["NROOM", "NBATH", "PATIO", "FIREPL", "AC", "GAR", "AGE", "LOTSZ", "SQFT"]
x = numpy.array([db.by_col(var) for var in x_names]).T

# Read spatial data
ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
w = ww.read()
ww.close()
w_name = "baltim_q.gal"
w.transform = "r"

model = spreg.GM_Lag(
    y,
    x,
    w=w,
    name_y=y_name,
    name_x=x_names,
    name_w="baltim_q",
    name_ds="baltim"
)
print(model.summary)



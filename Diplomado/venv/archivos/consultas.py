from pymongo import MongoClient
#PASO 1: conexion al servidor mongodb
mongoClient = MongoClient('localhost',27017)

#PASO 2: Conexion a la Base de Datos
db = mongoClient.diplomado
numero_15=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]

numero_Departamentos=["1","2","3","4","5","6","7","8","9","10","11","12","13",
                      "14","15","16","17","18","19","20","21","22","23","24",
                      "25","26","27","28","29","30","31","32"
                      ]

departamentos=["Amazonas","Antioquia","Arauca","Atlántico","Bolívar","Boyacá",
               "Caldas","Caquetá","Casanare","Cauca","Cesar","Chocó",
               "Cundinamarca","Córdoba","Guainía","Guaviare","La Guajira",
               "Huila","Magdalena","Meta","Nariño","Norte de Santander",
               "Putumayo","Quindío","Risaralda","San Andrés","Santander",
               "Sucre","Tolima","Valle del Cauca","Vaupés","Vichada"
               ]

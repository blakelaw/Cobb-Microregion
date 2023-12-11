from CentralCode import *

x = time.time()
boundary = [[34.223386, -84.281055],[34.498017, -84.303437],[34.514741, -84.613954],[34.210412, -84.606957]]
name = "NorthAtlanta"
Helper().mapcoordinate(boundary, name)
cleaned = Clean().fullycleanedf(boundary)
print("Final Time: ", time.time()-x)

fullycleaned = Clean().wordparse(cleaned, name,3)
unique = cleaned.drop_duplicates(subset=['Name', 'Label'])
markers = Helper().markers(boundary)
dictionary = Visualize().dictionary(fullycleaned, unique, 0.6, markers)
print(dictionary)
Visualize().visualizecities(dictionary, 3,10000, markers)

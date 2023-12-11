from CentralCode import *

x = time.time()
boundary = [[32.282663, -81.436383

],[32.134010, -80.731682],[31.537356, -81.139357

]]
name = "Savannah"
#Helper().mapcoordinate(boundary, name)
cleaned = Clean().fullycleanedf(boundary)
print("Final Time: ", time.time()-x)

fullycleaned = Clean().wordparse(cleaned, name,3, False)
unique = cleaned.drop_duplicates(subset=['Name', 'Label'])
markers = Helper().markers(boundary)
dictionary = Visualize().dictionary(fullycleaned, unique, 0.6, markers)

Visualize().visualizecities(dictionary, 3,10000, markers)

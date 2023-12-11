from CentralCode import *

x = time.time()
boundary = [[37.003978, -109.043078],[36.974306, -103.006659],[31.480609, -103.062380],[31.337942, -109.061652]]
name = "NorthAtlanta"
Helper().mapcoordinate(boundary, name)
cleaned = Clean().fullycleanedf(boundary)
print("Final Time: ", time.time()-x)

fullycleaned = Clean().wordparse(cleaned, name,3)
unique = cleaned.drop_duplicates(subset=['Name', 'Label'])
markers = Helper().markers(boundary)
dictionary = Visualize().dictionary(fullycleaned, unique, 0.6, markers)

Visualize().visualizecities(dictionary, 3,10000, markers)

#[[37.003978, -109.043078],[36.974306, -103.006659],[31.480609, -103.062380],[31.337942, -109.061652]]
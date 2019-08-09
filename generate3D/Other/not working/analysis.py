data = [1249.7, 1243.9, 1197.6, 1198.8, 1248.0, 1247.4, 1200.2, 1196.3, 1246.9, 1198.7, 1198.0, 1245.6, 1248.2, 1248.3, 1198.6, 1198.1, 1198.7, 1246.5, 1248.3, 1198.8, 1245.6, 1198.3, 1246.8, 1195.5, 1248.6, 1247.9, 1247.8, 1148.2, 1248.5, 1149.7, 1197.1, 1148.2, 1246.5, 1249.2, 1248.5, 1200.0, 1248.3, 1199.9, 1149.6, 1199.3, 1247.1, 1198.1, 1249.1, 1247.1, 1248.6, 1249.8, 1200.0, 1249.4, 1250.3, 1250.0, 1149.6, 1200.4, 1251.1, 1248.8, 1249.5, 1200.8, 1251.1, 1151.2, 1205.8, 1200.5, 1150.9, 1250.3, 1202.8, 1202.0, 1201.4, 1202.7, 1252.1, 1251.5, 1151.5, 1252.2, 1253.1, 1251.7, 1252.0, 1254.2, 1252.9, 1256.8, 1254.2, 1253.7, 1253.7, 1254.5, 1256.2, 1203.8, 1257.6, 1257.1, 1258.7, 1056.0, 1261.3, 1267.0, 1260.5, 1260.5]
print(max(data))
print(min(data))
print(max(data)-min(data))

total = 0
for i in data:
    total += i

average = total / len(data)
print(average)
print(average - min(data))
print(max(data)-average)
print()
varience = 0

for i in data:
    x = i - average
    x = x * x
    varience += x / len(data)

stdev = varience ** 0.5
print(stdev)


"""
30 pictures taken, at 0.04 intervals
saved variables in form: image()_r_ or image()_l_
Saved point clouds...
Distance to object: 1119.1 mm

30 pictures taken, at 0.04 intervals
saved variables in form: image()_r_ or image()_l_
Saved point clouds...
Distance to object: 1119.7 mm

30 pictures taken, at 0.04 intervals
saved variables in form: image()_r_ or image()_l_
Saved point clouds...
Distance to object: 1120.2 mm

30 pictures taken, at 0.04 intervals
saved variables in form: image()_r_ or image()_l_
Saved point clouds...
Distance to object: 1120.1 mm

30 pictures taken, at 0.04 intervals
saved variables in form: image()_r_ or image()_l_
Saved point clouds...
Distance to object: 1119.7 mm
"""

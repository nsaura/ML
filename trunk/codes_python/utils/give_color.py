# https://stackoverflow.com/questions/43001349/convert-numpy-array-of-rgb-values-to-hex-using-format-operator
rgb2hex = lambda r,g,b: '#%02x%02x%02x' %(r,g,b)
hex2rgb = lambda hexcode : tuple(map(ord,hexcode[1:].decode('hex')))

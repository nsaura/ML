import os 

## ## Change a name reccursively
for r, d, f in os.walk('./'):
    for n in f :
        m = n
        ## Try with a split to avoid to change the extension ".something" into "_something"
        m=m.replace(':','_')
        m=m.replace('.','_')
        
        os.system("mv %s %s" %(n, m))

## Change an extension reccursively
for r, d, f in os.walk('./'):
    for n in f :
        m = n[-3:]
        new_n = n[:-4] + '.' + m
        os.system("mv %s %s" %(n, new_n))



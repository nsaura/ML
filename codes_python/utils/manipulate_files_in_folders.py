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

# To premute names

for r, d, f in os.walk('./'):
    for n in f :
        if os.path.splitext(n)[1] == '.npy':
                 
            Nt_part = n[:6]
            Nx_part = n[6:12]
            other = n[12:]
            m = Nx_part + Nt_part + other
                 
            #print "original = ", n
            #print "modified = ", m
            
            os.system("mv %s %s" %(n, m))


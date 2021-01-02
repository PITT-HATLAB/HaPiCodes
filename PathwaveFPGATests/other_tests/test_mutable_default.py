def func(arg = [1,2,3]):
    print (arg)

func()
func(arg = [2,3,4,5])
func()


def append(element, seq=[]):
    seq.append(element)
    return seq

print(append(1))
print(append(2))

class person :
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address

    def greeting(self):
        print('Hello, my name is {0}.'.format(self.name))

# self -> class 
# coding=utf-8
import random

# classes = ['5', '6','9', '10','24', '35' , '37' , '40', '42', '20', '41','44','46', '61' , '63','64', '65','66', '67','72' , '74', '75' , '77','78' , '88' , '90' , '91', '93' , '104' ,  '105' ,'106']
classes = ['77', '72', '88', '105', '61', '104', '90', '67']

def main():
    global  classes
    for index in range(len(classes)):
            one = classes[index]
            print(one)
            list = classes[:index-1] + classes[index+1:]
            # slice = random.sample(list, 5)
            slice = random.sample(list, 1)
            print('one = %s, slice = %s' % (one, slice))


if __name__ == '__main__':
    main()

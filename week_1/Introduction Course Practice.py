# bubble sort

def sort(list):
    for num in range(len(list)-1, 0,-1):
        #num is an iterator like i which has the starting value of the range,
        # so that it runs all the passes needed to sort out the entire list
        for i in range(num):
            if list[i]>list[i+1]:
                temp= list[i]
                list[i]=list[i+1]
                list[i+1]=temp

list = [32,4,67,90,12,43,23,542,42,665,23,54,765,86.9,56,3,1,75,43]
sort(list)
print(list)
print("\n")

#Strings and manipulation of strings
fruit = "banana"
length = len(fruit)
print("The length is equal to", length)
welcome = "greetings on your introduction to this lecture on python. you and me makes we"

x= welcome.find('on')
print(x)
y= welcome.find('.')
print(y)
final = welcome[x:y]
print(final)
print("hope its back to normal")
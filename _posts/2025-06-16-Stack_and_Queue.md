---
layout: post
title: "Stack and Queue Implementation"
date: 2025-06-16
tags: [python]
categories: [DSA]
description: ""
featured: true
thumbnail: 
---

Here I plan to document the Stack and Queue Implementations in Python, these could be helpful to anyone preparing for the upcoming placement season. 

## Stack 

Simply put it's a linear data structure(linear data container) which holds data with a few quirky twist: 
1) Data can only be inserted from one end 
2) Inserted data can be removed from that very en

I know I made it a bit wordy, the famous 4 letter acronymn LIFO(Last-in First-out) principle nicely sums up the above jargon. 

Now Let's implement it in Python to see its beauty. 

```Python
class stack:
    def __init__(self,size):
        if size<=0 or type(size)==float:
            raise ValueError('Stack size must be a positive integer')
        self.size=size
        self.top=-1
        self.arr=[]

    def push(self,x):
        if self.top>=self.size-1:
            raise OverflowError('Stack is full')
        self.arr.append(x)
        self.top+=1
    
    def pop(self):
        if self.top==-1:
            raise IndexError('The Stack is empty, cannot pop')
        self.top-=1
        self.arr.pop()


    def peek(self):
        if self.top==-1:
            raise IndexError('The Stack is empty')
        return self.arr[self.top]

    def length(self):
        return self.top+1
```

### The Important Methods involved with Stack Data Structure are 
push, pop, peek and length. 

Push Method: As the word suggests, it pushes elements on to the stack from one end. 
- Time Complexity involved is O(1)

Pop Method: It pops the last element which was pushed onto the stack(LIFO is the Magic Mantra)
- Time Complexity involved is O(1)

Peek Method: This method enables us to take a quick peek at the stack. 

- Time Complexity involed is O(1)
  
Length Method: This method enables us to get the size of the stack. 


## Queue 


It's also a linear data structure which holds data in the following way: 
1) Data can only be inserted from one end 
2) Inserted data can be removed from the opposite end

The 4 letter acronymn which can come in handy is FIFO(First-in First-out)

Now Let's implement it in Python to better understand this data structure: 

```python
class Queue:
    def __init__(self,size):
        self.size=size
        self.arr=[None]*size
        self.current_size=0
        self.start,self.end=0,0

    def push(self,x):
        if (self.current_size==self.size):
            raise OverflowError("Queue is full")
        if self.current_size==0:
            self.start,self.end=0,0
        else:
            self.end=(self.end+1)%self.size
        self.arr[self.end]=x
        self.current_size+=1
    
    def pop(self):
        if self.current_size==0:
            raise IndexError("Queue is empty")
        a=self.arr[self.start]
        if self.current_size==1:
            self.start,self.end=-1,-1
        else:
            self.start=(self.start+1)%self.size
        self.current_size-=1
        return a

    def peek(self):
        if self.current_size==0:
            return IndexError("Queue is Empty")
        return self.arr[self.start]
    
    def length(self):
        return self.current_size
```

### The Important Methods involved with Queue Data Structure are 
push, pop, peek and length. 

Push Method: It pushes elements on to the queue from one end. 
- Time Complexity involved is O(1)

Pop Method: It pops the last element which was pushed first onto the queue 
- Time Complexity involved is O(1)

Peek Method: This method enables us to take a quick peek at the queue 

- Time Complexity involed is O(1)
  
Length Method: This method enables us to get the size of the queue. 

- Time Complexity involed is O(1)


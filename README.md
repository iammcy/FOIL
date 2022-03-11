# First Order Inductive Learner

This code is about using FOIL to realize rule inference for the given knowledge graph, and obtain new knowledge through "deduction". Note: this code was my work as a TA.

## Algorithm
1.  Construct positive examples, negative examples and background knowledge
2.  Predicates are successively added to inference rules as prerequisite constraint predicates
3.  The FOIL gain value of the new inference rule was calculated based on positive example, negative example and background knowledge
4.  The optimal premise constraint predicate was selected based on the calculated FOIL gain value
5.  If the inference rule covers positive examples in the training sample set, but does not cover any negative examples, the learning ends.

Note: The algorithm cannot derive rules for all cases

## Case 1
- **Input**
```cmd
5
Sibling(Ann,Mike)
Couple(David,James)
Mother(James,Ann)
Mother(James,Mike)
Father(David,Mike)
Father(x,y)
```

- **Output**
```cmd
Couple(x,z) Ʌ Mother(z,y) → Father(x,y)
Father(David,Ann)
```

## Case 2
- **Input**
```cmd
5
Father(Jack,Dell) 
Father(Dell,Stephen) 
Grandfather(Jack,Stephen) 
Father(Dell,Seth) 
Brother(Stephen,Seth)
Grandfather(x,y)
```

- **Output**
```cmd
Father(x,z) Ʌ Father(z,y) → Grandfather(x,y)
Grandfather(Jack,Seth)
```
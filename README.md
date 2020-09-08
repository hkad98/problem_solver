# Automatic Evaluation of Mathematical Word Problems
The main goal of this project was to create a program that would solve simple mathematical word problems in the Czech language for pupils of primary schools (1st to 3rd grade).

The program uses data extracted from corpus SYN2015.
https://wiki.korpus.cz/doku.php/:cnk:syn2015

Required libraries to run program are specified in file requirements.txt.

# Run

Unfortunately word problems licence does not allow us to make them freely accessible.
And therefore directory dataset is empty ans so is data/traindata.

## Dataset
500 Czech word problems were used in this program. Word problems from schoolbook [[1]](#1) were used.
Two datasets were created:
WP150 - containing 150 word problems
WP500 - containing 500 word problems

Word problem example:
```bash
Maminka koupila na trhu 15 kg třešní a 3 krát méně jahod. Kolik kg ovoce koupila dohromady? | 20 | NUM1 + NUM1 / NUM2
```


## References
<a id="1">[1]</a> 
Marie Reischigová.
Matematika na základní a obecné školeve slovních úlohách. 
Pansofia, 1996. In Czech.
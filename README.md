# YKar

YKar est un langage de programmation inspiré du lisp. C'est un langage axé fonctionnel et événementiel

# Exemples de syntaxes

```
(new twice (lambda ((x)) (* x 2)))
(new r 10)
(twice r)  ; 20
(new increment (lambda ((x 1) (i 1)) (+ x i)))
(increment)  ; 2
(increment (twice r))  ; 21
(say hello world ! how do you do ?)  ; hello world ! how do you do ?
(show r)  ; 10
(env)  ; ... print all th environement
(if (> r 11) (say r > 11) (say r <= 11))  ; r <= 11
(new-event my-event (= r 12) (say r = 12))
(set! r (increment r))  ; nothing is printed on the screen, r != 12
(set! r (increment r))  ; r = 12
(set r 1)  ; end of the event
(push my-event)  ; r = 12
(new var (symbol i'm a text))  ; r = 12 is not printed, was turned off after execution by push
(show var)  ; i'm a text
(del var)
(show var)  ; KeyError
(- 20 (+ 2 (* 3 (twice r))))  ; 42
```

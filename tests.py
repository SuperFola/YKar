# coding=utf-8

__author__ = 'Alexandre Plateau'
"""
Licence MIT
"""

from ykar import *


lis_tests = [
    ("(quote (testing 1 (2.0) -3.14e159))", "(\"testing\" 1 (2.0) -3.14e+159)"),
    ("(+ 2 2)", 4),
    ("(+ (* 2 100) (* 1 10))", 210),
    ("(if (> 6 5) (+ 1 1) (+ 2 2))", 2),
    ("(if (< 6 5) (+ 1 1) (+ 2 2))", 4),
    ("(new x 3)", None), ("(show x)", 3), ("(+ x x)", 6),
    ("(begin (set! x 1) (set! x (+ x 1)) (+ x 1))", 3),
    ("(new add (lambda (x 1) (+ x 1)))", None), ("(add)", 2), ("(add 2)", 3),
    ("(new increment (lambda (x 1) (k 1) (+ x k)))", None),
    ("(increment 10)", 11),
    ("(increment 20 22)", 42),
    ("(new kind (lambda x (+ x x)))", None),
    ("(kind 5)", 10),
    ("(new twice (lambda x (* 2 x)))", None), ("(twice 5)", 10),
    ("(new compose (lambda f g (lambda x (f (g x)))))", None),
    ("(new repeat (lambda f (compose f f)))", None),
    ("(new test (repeat twice))", None),
    ("(test 2)", 8),
    ("(new fact (lambda n (if (<= n 1) 1 (* n (fact (- n 1))))))", None),
    ("(fact 3)", 6),
    ("(fact 50)", 30414093201713378043612608166064768844377641568960512000000000000),
    ("(new abs2 (lambda n (if (> n 0) n (- 0 n))))", None),
    ("(list (abs2 -3) (abs2 0) (abs2 3))", [3, 0, 3]),
    ("""(new combine (lambda f
            (lambda x y
              (if (null? x) Nil
                  (f (list (car x) (car y))
                     ((combine f) (cdr x) (cdr y)))))))""", None),
    ("((combine cons) (list 1 2 3 4) (list 5 6 7 8))", [(1, 5), (2, 6), (3, 7), (4, 8)]),
    ("(new take (lambda n seq (if (<= n 0) 0 (cons (car seq) (take (- n 1) (cdr seq))))))", None),
    ("(new drop (lambda n seq (if (<= n 0) seq (drop (- n 1) (cdr seq)))))", None),
    ("(new mid (lambda seq (/ (length seq) 2)))", None),
    ("(new riff-shuffle (lambda deck ((combine append) (drop (mid deck) deck) (take (mid deck) deck))))", None),
    ("(riff-shuffle (list 1 2 3 4 5 6 7 8))", [1, 5, 2, 6, 3, 7, 4, 8]),
    ("(new f1 (open-output-file (quote fichier.txt)))", None),
    ("(write-in-file f1 (quote hello world !))", 13),
    ("(close-file f1)", None),
    ("(new f (open-input-file (quote fichier.txt)))", None),
    ("(new f3 (read-file f))", None),
    ("(show f3)", "hello world !"),
    ("(match f3 ((quote hello world) (- 0 1)) ((quote hello world  !) (+ 2 2)) ((quote hello world !) (+ 3 2)))", 5),
    ("(call/cc (lambda throw (+ 5 (* 10 (throw 1))))) ;; throw", 1),
    ("(call/cc (lambda throw (+ 5 (* 10 1)))) ;; do not throw", 15),
    ("""(call/cc (lambda throw
                 (+ 5 (* 10 (call/cc (lambda escape (* 100 (escape 3)))))))) ; 1 level""", 35),
    ("""(call/cc (lambda throw
                 (+ 5 (* 10 (call/cc (lambda escape (* 100 (throw 3)))))))) ; 2 levels""", 3),
    ("""(call/cc (lambda throw
                 (+ 5 (* 10 (call/cc (lambda escape (* 100 1))))))) ; 0 levels""", 1005),
    ("(new first car)", None),
    ("(new rest cdr)", None),
    ("(new count (lambda item L (if L (+ (equal? item (first L)) (count item (rest L))) 0)))", None),
    ("(count 0 (list 0 1 2 3 0 0))", 3),
    ("(count (quote the) (quote (the more the merrier the bigger the better)))", 4),
    ("((lambda n (+ n 1)) 4)", 5),
    ("""(new my-range (lambda start end
        (if (= start end)
            Nil
            (cons start (range (+ start 1) end)))))""", None),
    ("(my-range 1 5)", "(1 2 3 4)")
]
# ajouter les autres codes à tester (load-file)


def test(tests, name=''):
    """For each (exp, expected) test case, see if eval(parse(exp)) == expected."""
    fails = 0
    env = standard_env()
    for (x, expected) in tests:
        result = "nothing"
        try:
            result = eval_code(parse(x), env)
            # print(x, '=>', to_string(result))
        except Exception as e:
            print(x, '=raises=>', type(e).__name__, "->>", e)
            fails += 1
            print('FAIL!!!  Expected', expected, 'Got', result, "\n")
    print('%s %s: %d out of %d tests fail.' % ('*' * 45, name, fails, len(tests)))


if __name__ == '__main__':
    test(lis_tests, 'YKar')

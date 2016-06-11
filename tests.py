# coding=utf-8

__author__ = 'Alexandre Plateau'
"""
Licence MIT
"""

from ykar import *


lis_tests = [
    ("(say (testing 1 (2.0) -3.14e159))", "['testing', 1, [2.0], -3.14e+159]"),
    ("(new f1 (open-output-file (symbol fichier.txt)))", None),
    ("(write-in-file f1 (symbol hello world !))", 13),
    ("(close-file f1)", None),
    ("(new f (open-input-file (symbol fichier.txt)))", None),
    ("(read-file f)", "hello world !"),
    # ajouter les autres codes à tester (load-file)
    ("(+ 2 2)", 4),
    ("(+ (* 2 100) (* 1 10))", 210),
    ("(if (> 6 5) (+ 1 1) (+ 2 2))", 2),
    ("(if (< 6 5) (+ 1 1) (+ 2 2))", 4),
    ("(new x 3)", None), ("(show x)", 3), ("(+ x x)", 6),
    ("(begin (set! x 1) (set! x (+ x 1)) (+ x 1))", 3),
    ("(new add (lambda ((x 1)) (+ x 1)))", None), ("(add)", 2), ("(add 2)", 3),
    ("(new increment (lambda ((x 1) (k 1)) (+ x k)))", None),
    ("(increment 10)", 11),
    ("(increment 20 22)", 42),
    ("(new kind (lambda ((x)) (+ x x)))", None),
    ("(kind 5)", 10),
    ("(new twice (lambda ((x)) (* 2 x)))", None), ("(twice 5)", 10),
    ("(new compose (lambda ((f) (g)) (lambda (x) (f (g x)))))", None),
    ("(new repeat (lambda ((f)) (compose f f)))", None),
    ("(new fact (lambda ((n)) (if (<= n 1) 1 (* n (fact (- n 1))))))", None),
    ("(fact 3)", 6),
    ("(fact 50)", 30414093201713378043612608166064768844377641568960512000000000000),
    ("(new abs2 (lambda ((n)) (if (> n 0) n (- 0 n))))", None),
    ("(list (abs2 -3) (abs2 0) (abs2 3))", [3, 0, 3]),
    ("""(new combine (lambda ((f))
    (lambda ((x) (y))
      (if (null? x) (say ())
          (f (list (car x) (car y))
             ((combine f) (cdr x) (cdr y)))))))""", None),
    ("(new zip2 (combine cons))", None),
    ("(zip (list 1 2 3 4) (list 5 6 7 8))", [(1, 5), (2, 6), (3, 7), (4, 8)]),
    ("""(new riff-shuffle (lambda ((deck)) (begin
    (new take (lambda ((n) (seq)) (if (<= n 0) (say ()) (cons (car seq) (take (- n 1) (cdr seq))))))
    (new drop (lambda ((n) (seq)) (if (<= n 0) seq (drop (- n 1) (cdr seq)))))
    (new mid (lambda ((seq)) (/ (length seq) 2)))
    ((combine append) (take (mid deck) deck) (drop (mid deck) deck)))))""", None),
    ("(riff-shuffle (list 1 2 3 4 5 6 7 8))", [1, 5, 2, 6, 3, 7, 4, 8])
]


def test(tests, name=''):
    """For each (exp, expected) test case, see if eval(parse(exp)) == expected."""
    fails = 0
    env = standard_env()
    for (x, expected) in tests:
        result = "nothing"
        try:
            result = eval_code(parse(x), env)
            print(x, '=>', to_string(result))
            ok = (result == expected)
        except Exception as e:
            print(x, '=raises=>', type(e).__name__, e)
            ok = False
        if not ok:
            fails += 1
            print('FAIL!!!  Expected', expected, 'Got', result)
        print()
    print('%s %s: %d out of %d tests fail.' % ('*'*45, name, fails, len(tests)))


if __name__ == '__main__':
    test(lis_tests, 'YKar')

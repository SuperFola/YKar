__author__ = 'Folaefolc'
"""
Code par Alexandre Plateau
Licence MIT
"""

import math
import cmath
import operator as op
import time
import os
import itertools


start_token = '('
end_token = ')'
comment = ";"
block_comment_token = "/*"
block_comment_close_token = "*/"
namespace_tok = '::'
list_ns_opener = '{'
list_ns_closer = '}'
tokens = (start_token, end_token, namespace_tok, list_ns_opener, list_ns_closer)

language_name = 'YKar'

ev = {}
pushed = {}
is_checking_evs = False


def to_string(x):
    """Convert a Python object back into a Lisp-readable string."""
    if x is True:
        return "#t"
    elif x is False:
        return "#f"
    elif isinstance(x, str):
        return x
    elif isinstance(x, str):
        return '"%s"' % x.replace('"', r'\"')
    elif isinstance(x, list):
        return '(' + ' '.join(map(to_string, x)) + ')'
    elif isinstance(x, complex):
        return str(x).replace('j', 'i')
    else:
        return str(x)


def recur_join(sep, x):
    work = ""
    for e in x:
        if isinstance(e, list):
            work += str(e)
        else:
            work += str(e) + sep
    return work


class Env(dict):
    def __init__(self, parms=(), args=(), outer=None):
        super().__init__(self)
        self.outer = outer
        try:
            self.update(dict(parms))
            parms_keys = tuple(dict(parms).keys())
        except ValueError:
            parms_keys = tuple([i[0] for i in parms])
        self.update(zip(parms_keys, args))

    def __xor__(self, other):
        new = Env()
        for key in other:
            if key not in self:
                new[key] = other[key]
        return new

    def __and__(self, other):
        new = Env()
        for key in other:
            if key in self and key in other:
                new[key] = self[key]
        return new

    def __getitem__(self, var):
        return dict.__getitem__(self, var) if (var in self) else None

    def find(self, var):
        if var in self:
            return self[var]
        elif self.outer is not None:
            return self.outer.find(var)
        else:
            raise_error('KeyError', '\'{}\' doesn\'t exist (with type : {})'.format(var, type(var).__name__))
            return None


class Procedure(object):
    def __init__(self, params, body, envi):
        for elem in params:
            if not isinstance(elem, list):
                raise_error("SyntaxError", "Missing brace arround argument '%s'" % elem)
        self.params, self.body, self.env = params, body, envi

    def __call__(self, *args):
        return eval_code(self.body, Env(self.params, args, self.env))


def raise_error(err_type, msg):
    print(err_type, ':', msg)


def tokenize(chars):
    for tok in tokens:
        chars = chars.replace(tok, ' %s ' % tok)
    return chars.split()


def parse(program):
    tok = tokenize(program)
    if '(' and ')' in tok and len(tok) > 1:
        return read_from_tokens(tok)
    elif len(tok) == 1:
        return read_from_tokens(['('] + tok + [')'])
    return raise_error('SyntaxError', 'Missing the brackets')


def read_from_tokens(tokens):
    if not len(tokens):
        return raise_error('SyntaxError', 'Unexpected EOF while reading')

    token = tokens.pop(0)

    if token == block_comment_token:
        while tokens[0] != block_comment_close_token:
            tokens.pop(0)
        tokens.pop(0)
        token = tokens.pop(0)

    if token == start_token:
        ast = []
        while tokens[0] != end_token:
            ast.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return ast
    elif token == end_token:
        return raise_error('SyntaxError', 'Unexpected ' + token)
    elif token == comment:
        pass
    else:
        return atom(token)


def atom(token):
    if token == '#t':
        return True
    elif token == '#f':
        return False
    elif token[0] == '"':
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            try:
                return complex(token.replace('i', 'j', 1))
            except ValueError:
                return str(token)


def standard_env():
    env = Env()
    env.update(vars(math))  # sin, cos, sqrt, pi, ...
    env.update(vars(cmath))
    env.update({
        '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv, '//': op.floordiv, '%': op.mod,
        '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, '=': op.eq, '!=': op.ne,
        'not': op.not_, 'eq?': op.is_, 'equal?': op.eq,
        'abs': abs, 'zip': lambda *x: list(zip(*x)),
        'car': lambda x: x[0], 'cdr': lambda x: x[1:], 'cons': lambda x, y: [x] + y,
        'length': len, 'type': lambda x: type(x).__name__,
        'list': lambda *x: list(x), 'list?': lambda x: isinstance(x, list),
        'range': lambda start, stop: [_ for _ in range(start, stop + 1)],
        'map': lambda *x: list(map(*x)), 'max': max, 'min': min,
        'time': time.time, 'round': round,
        'open-input-file': open, 'open-output-file': lambda f: open(f, 'w'), 'close-file': lambda f: f.close(),
        'read-file': lambda f: f.read(), 'write-in-file': lambda f, s: f.write(s),
        'load-file': lambda f: load(f),
        'null': None, 'null?': lambda x: not not x,
        'int': lambda x: int(x), 'float': lambda x: float(x),
        'number?': lambda x: isinstance(x, (int, float)),
        'bool': lambda x: bool(x), 'bool?': lambda x: isinstance(x, bool),
        'procedure?': callable,
        'symbol?': lambda x: isinstance(x, str)
    })
    return env


def parse_use_instruction(env, code):
    full_ns = []
    list_content = []

    h_lnc = False
    h_lno = False

    for token in code:
        if token == list_ns_opener:
            h_lno = True
        elif token == list_ns_closer:
            h_lnc = True
        elif h_lnc and token not in (namespace_tok, list_ns_opener, list_ns_closer):
            raise_error('ParseError', 'Token \'%s\' was not expected' % token)

        if not h_lno and token != namespace_tok:
            full_ns.append(token)
            continue
        elif h_lno and not h_lnc and token != list_ns_opener:
            list_content.append(token)
            continue

    if os.path.exists(os.path.join(*full_ns)):
        if not list_content:
            env[namespace_tok.join(full_ns)] = load_env_from_file(os.path.join(*full_ns)) ^ standard_env()
        else:
            modules = Env()
            modules.update(zip(list_content, itertools.repeat(None)))
            env[namespace_tok.join(full_ns)] = load_env_from_file(os.path.join(*full_ns)) & modules
    else:
        raise_error('FileError', 'Lib can not be found at \'%s\'' % os.path.join(os.getcwd(), *full_ns))


def check_events():
    global ev, pushed, is_checking_evs, env

    if not is_checking_evs:
        is_checking_evs = True
        for name, (cond, body) in ev.items():
            if pushed[name]['triggered'] or eval_code(cond, env):
                tmp = eval_code(body, env)
                if isinstance(tmp, Procedure):
                    tmp(*[eval_code(a, env) for a in pushed[name]['args']])
                pushed[name]['triggered'] = False
    is_checking_evs = False


def eval_code(x, env):
    global ev

    if isinstance(x, list):
        if not x:
            return raise_error('RuntimeError', 'Empty statement')

    while True:
        check_events()

        if isinstance(x, str):
            return env.find(x)
        elif not isinstance(x, list):
            return x
        elif x[0] == "say":
            if len(x) >= 2:
                (_, *exp) = x
                return recur_join(' ', exp)
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need at least 1 argument")
        elif x[0] == "use":
            if len(x) >= 2:
                (_, *exp) = x
                parse_use_instruction(env, exp)
                return None
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need at least 1 argument")
        elif x[0] == "show":
            if len(x) == 2:
                (_, exp) = x
                return env.find(exp)
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need exactly 1 argument")
        elif x[0] == "match":
            if len(x) >= 3:
                (_, cond, *patterns) = x
                val = eval_code(cond, env)
                for (pattern, new_code) in patterns:
                    if val == eval_code(pattern, env):
                        x = eval_code(new_code, env)
            return raise_error("ArgumentError",
                               "'" + x[0] + "' need at least 2 arguments")
        elif x[0] == "lambda":
            if len(x) == 3:
                (_, params, body) = x
                return Procedure(params, body, env)
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need exactly 2 arguments")
        elif x[0] == "if":
            if len(x) == 4:
                (_, test, conseq, alt) = x
                x = conseq if eval_code(test, env) else alt
            if len(x) == 3:
                (_, test, conseq) = x
                if eval_code(test, env):
                    x = conseq
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need between 2 and 3 arguments")
        elif x[0] == "?":
            if len(x) == 2:
                (_, test) = x
                x = 1 if eval_code(test, env) else 0
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need exactly 1 argument")
        elif x[0] == "new":
            if len(x) == 3 or len(x) == 4 and x[2] in env.keys():
                (_, var, *exp) = x
                if var not in env.keys():
                    tmp = eval_code(exp[0], env)
                    if tmp:
                        env[var] = tmp
                        return None
                    else:
                        return raise_error('RuntimeError', 'The wanted value is impossible to create')
                else:
                    return raise_error("DefineError",
                                       "Can't override existing variable. Use set! instead")
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need exactly 2 arguments")
        elif x[0] == "new-event":
            if len(x) == 3:
                (_, var, cond, exp) = x
                if var not in ev.keys():
                    ev[var] = [eval_code(cond, env), eval_code(exp, env)]
                    pushed[var]['args'] = []
                    pushed[var]['triggered'] = False
                    return None
                else:
                    return raise_error("DefineError",
                                       "Can't override existing event-driven variable. Use set!-event instead")
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need exactly 2 arguments")
        elif x[0] == "set!":
            if len(x) == 3 or len(x) == 4 and x[2] in env.keys():
                (_, var, *exp) = x
                if var in env.keys():
                    tmp = eval_code(exp[0], env)
                    if tmp:
                        env[var] = tmp
                        return None
                    else:
                        return raise_error('RuntimeError', 'The wanted value is impossible to create')
                else:
                    return raise_error("SetError", "Can't overwrite a non existing variable. Use new instead")
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need exactly 2 arguments")
        elif x[0] == "set!-event":
            if len(x) == 3:
                (_, var, cond, exp) = x
                if var in ev.keys():
                    ev[var] = [eval_code(cond, env), eval_code(exp, env)]
                    pushed[var]['args'] = []
                    pushed[var]['triggered'] = False
                    return None
                else:
                    return raise_error("SetError",
                                       "Can't overwrite a non existing event-driven variable. Use new-event instead")
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need exactly 2 arguments")
        elif x[0] == "push":
            if len(x) >= 2:
                (_, var, args) = x
                if isinstance(ev[var], Procedure):
                    args = [eval_code(exp, env) for exp in args]
                    pushed[var]['args'] = args
                    pushed[var]['triggered'] = True
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need at least 2 arguments")
        elif x[0] == "del":
            if len(x) >= 2:
                (_, *exp) = x
                for e in exp:
                    if e in env.keys():
                        del env[e]
                    elif e in ev.keys():
                        del env[e]
                    else:
                        raise_error('ArgumentError', '\'%s\' does not exist' % e)
                return None
            else:
                return raise_error("ArgumentError",
                                   "'" + x[0] + "' need at least 1 argument")
        elif x[0] == "begin":
            for exp in x[1:]:
                eval_code(exp, env)
            x = x[-1]
        elif x[0] == "symbol":
            return recur_join(' ', x[1:])
        elif x[0] == "env":
            return sorted(list(env.keys()))
        else:
            if not isinstance(env.find(x[0]), str):
                exps = [eval_code(exp, env) for exp in x]
                proc = exps.pop(0)
                if isinstance(proc, Procedure):
                    x = proc.body
                    env = Env(proc.params, exps, proc.env)
                else:
                    try:
                        return proc(*exps)
                    except Exception as e:
                        if isinstance(e, TypeError):
                            return raise_error('RuntimeError', e)
                        return raise_error(type(e).__name__, e)


def load_env_from_file(path):
    with open(path) as sr:
        code = sr.readlines()

    _env = standard_env()
    for line in code:
        if line.count(start_token) == line.count(end_token) and line.strip()[:2] != comment:
            parsed = parse(line)
            eval_code(parsed, _env)
    del code
    return _env


def loop(env):
    std_prompt = language_name + ' > '
    not_eof_prompt = language_name + ' \' '

    prompt = std_prompt
    code = ""

    while True:
        code = input(prompt) if prompt != not_eof_prompt else code + " " + input(prompt)

        if code.count(start_token) != code.count(end_token):
            prompt = not_eof_prompt

            if code in env.keys():
                prompt = std_prompt

        if code.count(start_token) == code.count(end_token) and code.strip()[:2] != comment:
            prompt = std_prompt

            parsed = parse(code)
            val = eval_code(parsed, env)

            if val is not None:
                print(schemestr(val))


def load(file):
    try:
        with open(file) as f:
            for line in f.readline():
                if line.count(start_token) == line.count(end_token) and line.strip()[:2] != comment:
                    parsed = parse(line)
                    val = eval_code(parsed, env)

                    if val is not None:
                        print(schemestr(val))
    except OSError:
        raise_error('OSError', '\'%s\' does not exists' % file)


def schemestr(exp):
    return to_string(exp)


if __name__ == '__main__':
    t = [
        "db    db  db   dD   .d8b.   d8888b.", "`8b  d8'  88 ,8P'  d8' `8b  88  `8D",
        " `8bd8'   88,8P    88ooo88  88oobY'", "   88     88`8b    88~~~88  88`8b  ",
        "   88     88 `88.  88   88  88 `88.", "   YP     YP   YD  YP   YP  88   YD"
    ]
    i = ["Version 0.1", "Développé par Folaefolc", "", "", "", "(env) pour lister toutes les fonctions"]
    print("\n".join(" " * 4 + t[c] + " " * 8 + "- " + i[c] for c in range(len(t))), "\n")
    env = standard_env()
    loop(env)
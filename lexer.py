import ply.lex as lex

tokens = (
    "ID", "NUMBER", "TYPE", "STRING",
    "PLUS", "MINUS", "TIMES", "DIVIDE", "MATMUL", "POWER",
    "EQUALS", "PLUSEQ", "COLON", "COMMA", "ARROW",
    "LPAREN", "RPAREN",
    "LBRACKET", "RBRACKET",
    "NEWLINE", "INDENT", "DEDENT",
    "DEF", "RETURN", "FOR",
    "CLASS", "LAMBDA",
    "TANGENT",
    "IMAGINARY",
)

reserved = {
    "def": "DEF",
    "return": "RETURN",
    "for": "FOR",
    "class": "CLASS",
}

t_POWER    = r"\*\*"
t_PLUSEQ   = r"\+="
t_ARROW    = r"→"
t_PLUS     = r"\+"
t_MINUS    = r"-"
t_TIMES    = r"\*"
t_DIVIDE   = r"/"
t_MATMUL   = r"@"
t_EQUALS   = r"="
t_COLON    = r":"
t_COMMA    = r","

def t_LAMBDA(t):
    r"λ"
    return t
t_LPAREN   = r"\("
t_RPAREN   = r"\)"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_TANGENT  = r"T"

t_ignore = ""  # handle whitespace manually

def t_COMMENT(t):
    r"\#[^\n]*"
    pass  # Ignore comments

def t_STRING(t):
    r"'[^']*'|\"[^\"]*\""
    # Remove quotes from string
    t.value = t.value[1:-1]
    return t

def t_IMAGINARY(t):
    r"(?<![a-zA-Z0-9_])i(?![a-zA-Z0-9_])"
    return t

def t_TYPE(t):
    r"(ℝ|\\mathbb\{R\}|\\R|ℤ|ℕ|R(?![a-zA-Z0-9_])|Z(?![a-zA-Z0-9_])|N(?![a-zA-Z0-9_]))"
    if t.value in ("ℤ", "Z"):
        t.value = "ℤ"
    elif t.value in ("ℕ", "N"):
        t.value = "ℕ"
    else:
        t.value = "ℝ"
    return t

def t_NUMBER(t):
    r"\d+(\.\d+)?([eE][+-]?\d+)?"
    t.value = float(t.value)
    return t

def t_NEWLINE(t):
    r"\n[ \t]*"
    t.lexer.lineno += 1
    # Count leading spaces/tabs on next line (tabs = 4 spaces)
    indent_str = t.value[1:]  # Skip the newline itself
    indent = sum(4 if c == '\t' else 1 for c in indent_str)
    t.value = ("NEWLINE", indent)
    return t

def t_WHITESPACE(t):
    r"[ \t]+"
    pass  # Ignore whitespace within lines

def t_ID(t):
    r"[a-zA-Z_][a-zA-Z0-9_]*"
    t.type = reserved.get(t.value, "ID")
    return t

def t_error(t):
    raise SyntaxError(f"Illegal character '{t.value[0]}'")

_raw_lexer = lex.lex()


class IndentLexer:
    """Wrapper lexer that emits INDENT/DEDENT tokens based on indentation."""

    def __init__(self, lexer):
        self.lexer = lexer
        self.indent_stack = [0]  # Stack of indentation levels
        self.token_queue = []    # Queue for pending tokens
        self.pending_newline = None  # Store NEWLINE to emit before DEDENT
        self.after_for = False   # Track if we just saw FOR keyword
        self.bracket_depth = 0   # Track nesting of brackets/parens

    def input(self, data):
        # Ensure file ends with newline for proper DEDENT handling
        if not data.endswith('\n'):
            data += '\n'
        self.lexer.input(data)
        self.indent_stack = [0]
        self.token_queue = []
        self.pending_newline = None
        self.after_for = False
        self.bracket_depth = 0

    def token(self):
        # Return queued tokens first
        if self.token_queue:
            return self.token_queue.pop(0)

        tok = self.lexer.token()

        # After FOR, treat IMAGINARY as ID (allow 'i' as loop variable)
        if tok and tok.type == "FOR":
            self.after_for = True
        elif tok and self.after_for:
            if tok.type == "IMAGINARY":
                tok.type = "ID"
                tok.value = "i"
            self.after_for = False

        # Track bracket/parenthesis nesting
        if tok and tok.type in ("LPAREN", "LBRACKET"):
            self.bracket_depth += 1
        elif tok and tok.type in ("RPAREN", "RBRACKET"):
            self.bracket_depth -= 1

        if tok is None:
            # End of input - emit DEDENT for remaining indent levels
            if len(self.indent_stack) > 1:
                self.indent_stack.pop()
                dedent_tok = lex.LexToken()
                dedent_tok.type = "DEDENT"
                dedent_tok.value = None
                dedent_tok.lineno = self.lexer.lineno
                dedent_tok.lexpos = self.lexer.lexpos
                return dedent_tok
            return None

        if tok.type == "NEWLINE":
            # Skip newlines inside brackets/parens (like Python)
            if self.bracket_depth > 0:
                return self.token()  # Skip this newline, get next token

            _, new_indent = tok.value
            tok.value = 1  # Restore simple newline count

            current_indent = self.indent_stack[-1]

            if new_indent > current_indent:
                # Indentation increased - emit NEWLINE then INDENT
                self.indent_stack.append(new_indent)
                indent_tok = lex.LexToken()
                indent_tok.type = "INDENT"
                indent_tok.value = new_indent
                indent_tok.lineno = tok.lineno
                indent_tok.lexpos = tok.lexpos
                self.token_queue.append(indent_tok)
                return tok  # Return NEWLINE first

            elif new_indent < current_indent:
                # Indentation decreased - emit NEWLINE first, then DEDENT(s)
                # Queue DEDENTs
                while self.indent_stack and self.indent_stack[-1] > new_indent:
                    self.indent_stack.pop()
                    dedent_tok = lex.LexToken()
                    dedent_tok.type = "DEDENT"
                    dedent_tok.value = None
                    dedent_tok.lineno = tok.lineno
                    dedent_tok.lexpos = tok.lexpos
                    self.token_queue.append(dedent_tok)
                return tok  # Return NEWLINE first, DEDENTs are queued

        return tok

lexer = IndentLexer(_raw_lexer)

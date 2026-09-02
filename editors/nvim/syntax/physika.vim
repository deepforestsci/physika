if exists("b:current_syntax")
  finish
endif

syntax case match

syntax keyword physikaTodo TODO FIXME XXX NOTE contained
syntax match physikaComment "#.*$" contains=physikaTodo,@Spell

syntax region physikaString start=+'+ end=+'+ oneline contains=@Spell
syntax region physikaString start=+"+ end=+"+ oneline contains=@Spell

syntax match physikaNumber "\<\d\+\(\.\d\+\)\=\([eE][+-]\=\d\+\)\=\>"
syntax match physikaComplex "\<\d\+\(\.\d\+\)\=j\>"
syntax match physikaImaginary "\<i\>"

syntax keyword physikaConditional if else
syntax keyword physikaRepeat for
syntax keyword physikaInclude from import
syntax keyword physikaStructure class
syntax keyword physikaStatement return
syntax keyword physikaDefine def

syntax keyword physikaThis this

syntax match physikaType "[ℝℤℕℂ]"
syntax keyword physikaType R Z N
syntax match physikaType "\\mathbb{R}"
syntax match physikaType "\\mathbb{Z}"
syntax match physikaType "\\mathbb{N}"
syntax match physikaType "\\mathbb{C}"
syntax match physikaType "\\R\>"
syntax match physikaType "\\Z\>"
syntax match physikaType "\\N\>"

syntax keyword physikaType Symbol Function Equation

syntax keyword physikaBuiltinFunc exp log sin cos sqrt abs sum mean real
syntax keyword physikaBuiltinFunc floor erfc roll zeros gelu gt le arange
syntax keyword physikaBuiltinFunc mod mask_select masked_scatter concat
syntax keyword physikaBuiltinFunc reshape eigvalsh eigvecsh diag_embed
syntax keyword physikaBuiltinFunc fft ifft fft2 ifft2 fftn ifftn rfft irfft
syntax keyword physikaBuiltinFunc grad diff subs lambdify symbolic_solve

syntax match physikaFuncName "\%(\<def\s\+\)\@<=\k\+\ze\s*("

syntax match physikaClassName "\%(\<class\s\+\)\@<=\k\+"

syntax match physikaLambdaDef "\%(\<def\s*\)\@<=λ\ze\s*("

syntax match physikaOperator "\*\*\|+=\|==\|!=\|<=\|>=\|//\|[-+*/@<>=]"
syntax match physikaOperator "→\|->"
syntax match physikaColon ":"
syntax match physikaOperator ":="
syntax match physikaDelimiter "[()\[\],]"

highlight default link physikaComment      Comment
highlight default link physikaTodo         Todo
highlight default link physikaString       String
highlight default link physikaNumber       Number
highlight default link physikaComplex      Number
highlight default link physikaImaginary    Number
highlight default link physikaConditional  Conditional
highlight default link physikaRepeat       Repeat
highlight default link physikaInclude      Include
highlight default link physikaStructure    Structure
highlight default link physikaStatement    Statement
highlight default link physikaDefine       Keyword
highlight default link physikaLambdaDef    Keyword
highlight default link physikaThis         Special
highlight default link physikaType         Type
highlight default link physikaBuiltinFunc  Function
highlight default link physikaFuncName     Function
highlight default link physikaClassName    Type
highlight default link physikaOperator     Operator
highlight default link physikaColon        Operator
highlight default link physikaDelimiter    Delimiter

let b:current_syntax = "physika"

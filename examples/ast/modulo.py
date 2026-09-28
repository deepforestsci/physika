EXPECTED = {
    'functions': {},
    'classes': {},
    'program':
  (
    'expr',
    (
      'call',
      'print',
      [
        (
          'mod',
          ('num', 10),
          ('num', 2),
        ),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'call',
      'print',
      [
        (
          'mod',
          ('num', 12),
          ('num', 5),
        ),
      ],
    ),
    0,
  )
  (
    'decl',
    'a',
    'ℝ',
    ('num', 10),
    10,
  )
  (
    'decl',
    'b',
    'ℝ',
    ('num', 4),
    11,
  )
  (
    'expr',
    (
      'call',
      'print',
      [
        (
          'mod',
          ('var', 'a'),
          ('var', 'b'),
        ),
      ],
    ),
    0,
  )
  (
    'decl',
    'a',
    (
      'tensor',
      [
        (4, 'invariant'),
      ],
    ),
    (
      'array',
      [
        ('num', 10),
        ('num', 12),
        ('num', 15),
        ('num', 17),
      ],
    ),
    17,
  )
  (
    'decl',
    'b',
    'ℝ',
    ('num', 4),
    18,
  )
  (
    'expr',
    (
      'call',
      'print',
      [
        (
          'mod',
          ('var', 'a'),
          ('var', 'b'),
        ),
      ],
    ),
    0,
  )
  (
    'decl',
    'a',
    (
      'tensor',
      [
        (4, 'invariant'),
      ],
    ),
    (
      'array',
      [
        ('num', 10),
        ('num', 12),
        ('num', 15),
        ('num', 17),
      ],
    ),
    24,
  )
  (
    'decl',
    'b',
    (
      'tensor',
      [
        (4, 'invariant'),
      ],
    ),
    (
      'array',
      [
        ('num', 1),
        ('num', 3),
        ('num', 5),
        ('num', 7),
      ],
    ),
    25,
  )
  (
    'expr',
    (
      'call',
      'print',
      [
        (
          'mod',
          ('var', 'a'),
          ('var', 'b'),
        ),
      ],
    ),
    0,
  )
}

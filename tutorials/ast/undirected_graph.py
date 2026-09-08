=== UNIFIED AST ===
Functions:
  empty_graph:
    params: [('n_vertices', 'ℕ')]
    statements:
      (
        'body_decl',
        'z',
        (
          'tensor',
          [
            ('n_vertices', 'invariant'),
            ('n_vertices', 'invariant'),
          ],
        ),
        (
          'for_expr',
          'a',
          ('var', 'n_vertices'),
          (
            'for_expr',
            'b',
            ('var', 'n_vertices'),
            (
              'mul',
              (
                'add',
                ('var', 'a'),
                ('var', 'b'),
              ),
              ('num', 0.0),
            ),
          ),
        ),
      )
      (
        'body_decl',
        'g',
        ('struct_type', 'UndirectedGraph'),
        (
          'call',
          'UndirectedGraph',
          [],
        ),
      )
      (
        'body_field_assign',
        ('var', 'g'),
        'adjacency',
        ('var', 'z'),
      )
    body:
      ('var', 'g')

Classes:
  UndirectedGraph:
    class_params: []
    class_fields: [
  (
    'adjacency',
    (
      'tensor',
      [
        ('n', 'invariant'),
        ('n', 'invariant'),
      ],
    ),
  ),
]
    class_methods: [
  {
    name: 'num_vertices'
    params: []
    return_type: 'ℝ'
    statements: []
    body:
      (
        'mul',
        (
          'call',
          'len',
          [
            (
              'field_access',
              ('var', 'this'),
              'adjacency',
            ),
          ],
        ),
        ('num', 1.0),
      )
  },
  {
    name: 'has_edge'
    params:
      [
        ('u', 'ℝ'),
        ('v', 'ℝ'),
      ]
    return_type: 'ℝ'
    statements:
      [
        (
          'body_decl',
          'm',
          (
            'tensor',
            [
              ('n', 'invariant'),
              ('n', 'invariant'),
            ],
          ),
          (
            'field_access',
            ('var', 'this'),
            'adjacency',
          ),
        ),
        (
          'body_decl',
          'r',
          (
            'tensor',
            [
              ('n', 'invariant'),
            ],
          ),
          (
            'index',
            'm',
            ('var', 'u'),
          ),
        ),
      ]
    body:
      (
        'index',
        'r',
        ('var', 'v'),
      )
  },
  {
    name: 'degree'
    params:
      [
        ('u', 'ℝ'),
      ]
    return_type: 'ℝ'
    statements:
      [
        (
          'body_decl',
          'm',
          (
            'tensor',
            [
              ('n', 'invariant'),
              ('n', 'invariant'),
            ],
          ),
          (
            'field_access',
            ('var', 'this'),
            'adjacency',
          ),
        ),
        (
          'body_decl',
          'r',
          (
            'tensor',
            [
              ('n', 'invariant'),
            ],
          ),
          (
            'index',
            'm',
            ('var', 'u'),
          ),
        ),
      ]
    body:
      (
        'call',
        'sum',
        [
          ('var', 'r'),
        ],
      )
  },
  {
    name: 'neighbors'
    params:
      [
        ('u', 'ℝ'),
      ]
    return_type:
      (
        'tensor',
        [
          ('n', 'invariant'),
        ],
      )
    statements:
      [
        (
          'body_decl',
          'm',
          (
            'tensor',
            [
              ('n', 'invariant'),
              ('n', 'invariant'),
            ],
          ),
          (
            'field_access',
            ('var', 'this'),
            'adjacency',
          ),
        ),
      ]
    body:
      (
        'index',
        'm',
        ('var', 'u'),
      )
  },
  {
    name: 'add_edge'
    params:
      [
        ('u', 'ℝ'),
        ('v', 'ℝ'),
      ]
    return_type: None
    statements:
      [
        (
          'body_decl',
          'm',
          (
            'tensor',
            [
              ('n', 'invariant'),
              ('n', 'invariant'),
            ],
          ),
          (
            'field_access',
            ('var', 'this'),
            'adjacency',
          ),
        ),
        (
          'body_decl',
          'k',
          'ℕ',
          (
            'call',
            'len',
            [
              ('var', 'm'),
            ],
          ),
        ),
        (
          'body_decl',
          'new_adj',
          (
            'tensor',
            [
              ('n', 'invariant'),
              ('n', 'invariant'),
            ],
          ),
          (
            'for_expr',
            'a',
            ('var', 'k'),
            (
              'for_expr',
              'b',
              ('var', 'k'),
              (
                'indexN',
                'm',
                [
                  (
                    'index_item',
                    ('var', 'a'),
                  ),
                  (
                    'index_item',
                    ('var', 'b'),
                  ),
                ],
              ),
            ),
          ),
        ),
        (
          'body_index_assign_nd',
          'new_adj',
          [
            (
              'index_item',
              ('var', 'u'),
            ),
            (
              'index_item',
              ('var', 'v'),
            ),
          ],
          ('num', 1.0),
        ),
        (
          'body_index_assign_nd',
          'new_adj',
          [
            (
              'index_item',
              ('var', 'v'),
            ),
            (
              'index_item',
              ('var', 'u'),
            ),
          ],
          ('num', 1.0),
        ),
        (
          'body_field_assign',
          ('var', 'this'),
          'adjacency',
          ('var', 'new_adj'),
        ),
      ]
    body: None
  },
  {
    name: 'grow_adjacency'
    params:
      [
        ('new_n', 'ℕ'),
      ]
    return_type:
      (
        'tensor',
        [
          ('new_n', 'invariant'),
          ('new_n', 'invariant'),
        ],
      )
    statements:
      [
        (
          'body_decl',
          'old',
          (
            'tensor',
            [
              ('n', 'invariant'),
              ('n', 'invariant'),
            ],
          ),
          (
            'field_access',
            ('var', 'this'),
            'adjacency',
          ),
        ),
        (
          'body_decl',
          'result',
          (
            'tensor',
            [
              ('new_n', 'invariant'),
              ('new_n', 'invariant'),
            ],
          ),
          (
            'for_expr',
            'a',
            ('var', 'new_n'),
            (
              'for_expr',
              'b',
              ('var', 'new_n'),
              (
                'mul',
                (
                  'add',
                  ('var', 'a'),
                  ('var', 'b'),
                ),
                ('num', 0.0),
              ),
            ),
          ),
        ),
        (
          'body_decl',
          'm',
          'ℕ',
          (
            'call',
            'len',
            [
              ('var', 'old'),
            ],
          ),
        ),
        (
          'body_for_range',
          'a',
          ('num', 0),
          ('var', 'm'),
          [
            (
              'loop_for_range',
              'b',
              ('num', 0),
              ('var', 'm'),
              [
                (
                  'loop_index_assign_nd',
                  'result',
                  [
                    (
                      'index_item',
                      ('var', 'a'),
                    ),
                    (
                      'index_item',
                      ('var', 'b'),
                    ),
                  ],
                  (
                    'indexN',
                    'old',
                    [
                      (
                        'index_item',
                        ('var', 'a'),
                      ),
                      (
                        'index_item',
                        ('var', 'b'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ]
    body: ('var', 'result')
  },
  {
    name: 'add_vertex'
    params:
      [
        ('new_n', 'ℕ'),
      ]
    return_type: None
    statements:
      [
        (
          'body_field_assign',
          ('var', 'this'),
          'adjacency',
          (
            'method_call',
            ('var', 'this'),
            'grow_adjacency',
            [
              ('var', 'new_n'),
            ],
          ),
        ),
      ]
    body: None
  },
]

Program:
  ('class_def', 'UndirectedGraph')
  ('func_def', 'empty_graph')
  (
    'decl',
    'n0',
    'ℕ',
    ('num', 3),
    40,
  )
  (
    'assign',
    'g',
    (
      'call',
      'empty_graph',
      [
        ('var', 'n0'),
      ],
    ),
    41,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'num_vertices',
      [],
    ),
    0,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'add_edge',
      [
        ('num', 0.0),
        ('num', 1.0),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'add_edge',
      [
        ('num', 1.0),
        ('num', 2.0),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'neighbors',
      [
        ('num', 1.0),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'degree',
      [
        ('num', 1.0),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'has_edge',
      [
        ('num', 0.0),
        ('num', 2.0),
      ],
    ),
    0,
  )
  (
    'decl',
    'n3',
    'ℕ',
    ('num', 4),
    51,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'add_vertex',
      [
        ('var', 'n3'),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'add_edge',
      [
        ('num', 2.0),
        ('num', 3.0),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'method_call',
      ('var', 'g'),
      'degree',
      [
        ('num', 3.0),
      ],
    ),
    0,
  )
  ✓ No type errors found
3.0 ∈ ℝ
[1.0, 0.0, 1.0] ∈ ℝ[3]
2.0 ∈ ℝ
0.0 ∈ ℝ
1.0 ∈ ℝ

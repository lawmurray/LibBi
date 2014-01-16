####################################################################
#
#    This file was generated using Parse::Yapp version 1.05.
#
#        Don't edit this file, use source file instead.
#
#             ANY CHANGE MADE HERE WILL BE LOST !
#
####################################################################
package Parse::Bi;
use vars qw ( @ISA );
use strict;

@ISA= qw ( Parse::Yapp::Driver );
use Parse::Yapp::Driver;



sub new {
        my($class)=shift;
        ref($class)
    and $class=ref($class);

    my($self)=$class->SUPER::new( yyversion => '1.05',
                                  yystates =>
[
	{#State 0
		ACTIONS => {
			'MODEL' => 1
		},
		GOTOS => {
			'model' => 2
		}
	},
	{#State 1
		ACTIONS => {
			'IDENTIFIER' => 3
		},
		GOTOS => {
			'spec' => 4
		}
	},
	{#State 2
		ACTIONS => {
			'' => 5
		}
	},
	{#State 3
		ACTIONS => {
			"(" => 6
		},
		DEFAULT => -9
	},
	{#State 4
		ACTIONS => {
			"{" => 7
		},
		GOTOS => {
			'model_start' => 8
		}
	},
	{#State 5
		DEFAULT => 0
	},
	{#State 6
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 24,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37,
			")" => 21
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'positional_args' => 23,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'positional_arg' => 28,
			'postfix_expression' => 29,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'named_args' => 36,
			'cast_expression' => 38
		}
	},
	{#State 7
		DEFAULT => -3
	},
	{#State 8
		ACTIONS => {
			'NOISE' => 53,
			"}" => 39,
			'INPUT' => 40,
			'SUB' => 55,
			'CONST' => 57,
			'INLINE' => 56,
			'PARAM_AUX' => 43,
			'DIM' => 60,
			'STATE' => 61,
			'OBS' => 48,
			'PARAM' => 63,
			'STATE_AUX' => 52
		},
		GOTOS => {
			'inline' => 54,
			'input' => 41,
			'model_end' => 42,
			'state' => 44,
			'model_definitions' => 58,
			'param_aux' => 45,
			'const' => 59,
			'top_level' => 46,
			'var' => 47,
			'obs' => 62,
			'dim' => 49,
			'model_definition' => 50,
			'param' => 51,
			'state_aux' => 65,
			'noise' => 64
		}
	},
	{#State 9
		DEFAULT => -144
	},
	{#State 10
		DEFAULT => -143
	},
	{#State 11
		DEFAULT => -179
	},
	{#State 12
		DEFAULT => -132
	},
	{#State 13
		DEFAULT => -131
	},
	{#State 14
		DEFAULT => -171
	},
	{#State 15
		DEFAULT => -173
	},
	{#State 16
		DEFAULT => -145
	},
	{#State 17
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -150
	},
	{#State 18
		DEFAULT => -18
	},
	{#State 19
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 69,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 20
		DEFAULT => -146
	},
	{#State 21
		DEFAULT => -8
	},
	{#State 22
		ACTIONS => {
			'EQ_OP' => 70,
			'NE_OP' => 71
		},
		DEFAULT => -170
	},
	{#State 23
		ACTIONS => {
			"," => 72,
			")" => 73
		}
	},
	{#State 24
		ACTIONS => {
			"[" => 74,
			"=" => 75,
			"(" => 76
		},
		DEFAULT => -134
	},
	{#State 25
		DEFAULT => -162
	},
	{#State 26
		ACTIONS => {
			"?" => 77,
			'OR_OP' => 78
		},
		DEFAULT => -177
	},
	{#State 27
		ACTIONS => {
			"-" => 79,
			'ELEM_ADD_OP' => 82,
			"+" => 80,
			'ELEM_SUB_OP' => 81
		},
		DEFAULT => -161
	},
	{#State 28
		DEFAULT => -17
	},
	{#State 29
		DEFAULT => -141
	},
	{#State 30
		DEFAULT => -20
	},
	{#State 31
		DEFAULT => -172
	},
	{#State 32
		ACTIONS => {
			"<" => 83,
			'LE_OP' => 84,
			'GE_OP' => 85,
			">" => 86
		},
		DEFAULT => -167
	},
	{#State 33
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 87,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'cast_expression' => 38
		}
	},
	{#State 34
		ACTIONS => {
			'AND_OP' => 88
		},
		DEFAULT => -175
	},
	{#State 35
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -156
	},
	{#State 36
		ACTIONS => {
			"," => 94,
			")" => 95
		}
	},
	{#State 37
		DEFAULT => -133
	},
	{#State 38
		DEFAULT => -147
	},
	{#State 39
		DEFAULT => -4
	},
	{#State 40
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 99,
			'input_declarations' => 100,
			'input_declaration' => 96
		}
	},
	{#State 41
		DEFAULT => -47
	},
	{#State 42
		DEFAULT => -2
	},
	{#State 43
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 101,
			'param_aux_declaration' => 102,
			'param_aux_declarations' => 103
		}
	},
	{#State 44
		DEFAULT => -44
	},
	{#State 45
		DEFAULT => -50
	},
	{#State 46
		DEFAULT => -28
	},
	{#State 47
		DEFAULT => -25
	},
	{#State 48
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 106,
			'obs_declarations' => 104,
			'obs_declaration' => 105
		}
	},
	{#State 49
		DEFAULT => -24
	},
	{#State 50
		DEFAULT => -23
	},
	{#State 51
		DEFAULT => -49
	},
	{#State 52
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 109,
			'state_aux_declarations' => 107,
			'state_aux_declaration' => 108
		}
	},
	{#State 53
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 112,
			'noise_declarations' => 111,
			'noise_declaration' => 110
		}
	},
	{#State 54
		DEFAULT => -27
	},
	{#State 55
		ACTIONS => {
			'IDENTIFIER' => 3,
			"{" => 115
		},
		GOTOS => {
			'spec' => 114,
			'block_start' => 113,
			'block' => 116
		}
	},
	{#State 56
		ACTIONS => {
			'IDENTIFIER' => 118
		},
		GOTOS => {
			'inline_declaration' => 117,
			'inline_declarations' => 119
		}
	},
	{#State 57
		ACTIONS => {
			'IDENTIFIER' => 122
		},
		GOTOS => {
			'const_declarations' => 120,
			'const_declaration' => 121
		}
	},
	{#State 58
		ACTIONS => {
			'NOISE' => 53,
			"}" => 39,
			'INPUT' => 40,
			'SUB' => 55,
			'CONST' => 57,
			'INLINE' => 56,
			'PARAM_AUX' => 43,
			'DIM' => 60,
			'STATE' => 61,
			'OBS' => 48,
			'PARAM' => 63,
			'STATE_AUX' => 52
		},
		GOTOS => {
			'inline' => 54,
			'input' => 41,
			'model_end' => 123,
			'state' => 44,
			'param_aux' => 45,
			'const' => 59,
			'top_level' => 46,
			'var' => 47,
			'obs' => 62,
			'dim' => 49,
			'model_definition' => 124,
			'param' => 51,
			'state_aux' => 65,
			'noise' => 64
		}
	},
	{#State 59
		DEFAULT => -26
	},
	{#State 60
		ACTIONS => {
			'IDENTIFIER' => 3
		},
		GOTOS => {
			'spec' => 126,
			'dim_declarations' => 125,
			'dim_declaration' => 127
		}
	},
	{#State 61
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'state_declarations' => 130,
			'array_spec' => 129,
			'state_declaration' => 128
		}
	},
	{#State 62
		DEFAULT => -48
	},
	{#State 63
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 131,
			'param_declaration' => 132,
			'param_declarations' => 133
		}
	},
	{#State 64
		DEFAULT => -46
	},
	{#State 65
		DEFAULT => -45
	},
	{#State 66
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 134,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 67
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 135,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 68
		ACTIONS => {
			"[" => 74,
			"(" => 76
		},
		DEFAULT => -134
	},
	{#State 69
		DEFAULT => -142
	},
	{#State 70
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'relational_expression' => 136,
			'pow_expression' => 17,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 71
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'relational_expression' => 137,
			'pow_expression' => 17,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 72
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 24,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 138,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'named_args' => 139,
			'cast_expression' => 38
		}
	},
	{#State 73
		DEFAULT => -6
	},
	{#State 74
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 140,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'index_args' => 141,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'index_arg' => 142,
			'cast_expression' => 38
		}
	},
	{#State 75
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 143,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'cast_expression' => 38
		}
	},
	{#State 76
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 24,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37,
			")" => 144
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'positional_args' => 145,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 28,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'named_args' => 146,
			'cast_expression' => 38
		}
	},
	{#State 77
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 147,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'cast_expression' => 38
		}
	},
	{#State 78
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'equality_expression' => 22,
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'pow_expression' => 17,
			'logical_and_expression' => 148,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 79
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 17,
			'multiplicative_expression' => 149,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 80
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 17,
			'multiplicative_expression' => 150,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 81
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 17,
			'multiplicative_expression' => 151,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 82
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 17,
			'multiplicative_expression' => 152,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 83
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'shift_expression' => 153,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'pow_expression' => 17,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'cast_expression' => 38
		}
	},
	{#State 84
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'shift_expression' => 154,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'pow_expression' => 17,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'cast_expression' => 38
		}
	},
	{#State 85
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'shift_expression' => 155,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'pow_expression' => 17,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'cast_expression' => 38
		}
	},
	{#State 86
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'shift_expression' => 156,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'pow_expression' => 17,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'cast_expression' => 38
		}
	},
	{#State 87
		ACTIONS => {
			")" => 157
		}
	},
	{#State 88
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'equality_expression' => 22,
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'and_expression' => 14,
			'inclusive_or_expression' => 158,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'pow_expression' => 17,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 89
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 159,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 90
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 160,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 91
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 161,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 92
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 162,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 93
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'pow_expression' => 163,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 94
		ACTIONS => {
			'IDENTIFIER' => 164
		},
		GOTOS => {
			'named_arg' => 165
		}
	},
	{#State 95
		DEFAULT => -7
	},
	{#State 96
		DEFAULT => -68
	},
	{#State 97
		DEFAULT => -10
	},
	{#State 98
		ACTIONS => {
			"[" => 166,
			"(" => 6
		},
		DEFAULT => -9
	},
	{#State 99
		ACTIONS => {
			";" => 167
		},
		DEFAULT => -70
	},
	{#State 100
		ACTIONS => {
			"," => 168
		},
		DEFAULT => -66
	},
	{#State 101
		ACTIONS => {
			";" => 169
		},
		DEFAULT => -85
	},
	{#State 102
		DEFAULT => -83
	},
	{#State 103
		ACTIONS => {
			"," => 170
		},
		DEFAULT => -81
	},
	{#State 104
		ACTIONS => {
			"," => 171
		},
		DEFAULT => -71
	},
	{#State 105
		DEFAULT => -73
	},
	{#State 106
		ACTIONS => {
			";" => 172
		},
		DEFAULT => -75
	},
	{#State 107
		ACTIONS => {
			"," => 173
		},
		DEFAULT => -56
	},
	{#State 108
		DEFAULT => -58
	},
	{#State 109
		ACTIONS => {
			";" => 174
		},
		DEFAULT => -60
	},
	{#State 110
		DEFAULT => -63
	},
	{#State 111
		ACTIONS => {
			"," => 175
		},
		DEFAULT => -61
	},
	{#State 112
		ACTIONS => {
			";" => 176
		},
		DEFAULT => -65
	},
	{#State 113
		ACTIONS => {
			'NOISE' => 53,
			"}" => 177,
			'IDENTIFIER' => 183,
			'INPUT' => 40,
			"{" => 115,
			'INLINE' => 56,
			'CONST' => 57,
			'PARAM_AUX' => 43,
			'DIM' => 60,
			'STATE' => 61,
			'OBS' => 48,
			'DO' => 194,
			'PARAM' => 63,
			'STATE_AUX' => 52
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 178,
			'block_start' => 113,
			'do' => 187,
			'set_to' => 188,
			'state' => 44,
			'param_aux' => 45,
			'target' => 179,
			'var' => 189,
			'dim' => 180,
			'param' => 51,
			'block_definitions' => 181,
			'inline' => 182,
			'spec' => 114,
			'block_definition' => 184,
			'dtarget' => 185,
			'const' => 186,
			'action' => 191,
			'block_end' => 190,
			'obs' => 62,
			'block' => 192,
			'varies_as' => 193,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 114
		ACTIONS => {
			"{" => 115
		},
		GOTOS => {
			'block_start' => 195
		}
	},
	{#State 115
		DEFAULT => -107
	},
	{#State 116
		DEFAULT => -100
	},
	{#State 117
		DEFAULT => -41
	},
	{#State 118
		ACTIONS => {
			"=" => 196
		}
	},
	{#State 119
		ACTIONS => {
			"," => 197
		},
		DEFAULT => -39
	},
	{#State 120
		ACTIONS => {
			"," => 198
		},
		DEFAULT => -34
	},
	{#State 121
		DEFAULT => -36
	},
	{#State 122
		ACTIONS => {
			"=" => 199
		}
	},
	{#State 123
		DEFAULT => -1
	},
	{#State 124
		DEFAULT => -22
	},
	{#State 125
		ACTIONS => {
			"," => 200
		},
		DEFAULT => -29
	},
	{#State 126
		ACTIONS => {
			";" => 201
		},
		DEFAULT => -33
	},
	{#State 127
		DEFAULT => -31
	},
	{#State 128
		DEFAULT => -53
	},
	{#State 129
		ACTIONS => {
			";" => 202
		},
		DEFAULT => -55
	},
	{#State 130
		ACTIONS => {
			"," => 203
		},
		DEFAULT => -51
	},
	{#State 131
		ACTIONS => {
			";" => 204
		},
		DEFAULT => -80
	},
	{#State 132
		DEFAULT => -78
	},
	{#State 133
		ACTIONS => {
			"," => 205
		},
		DEFAULT => -76
	},
	{#State 134
		DEFAULT => -149
	},
	{#State 135
		DEFAULT => -148
	},
	{#State 136
		ACTIONS => {
			"<" => 83,
			'LE_OP' => 84,
			'GE_OP' => 85,
			">" => 86
		},
		DEFAULT => -168
	},
	{#State 137
		ACTIONS => {
			"<" => 83,
			'LE_OP' => 84,
			'GE_OP' => 85,
			">" => 86
		},
		DEFAULT => -169
	},
	{#State 138
		DEFAULT => -16
	},
	{#State 139
		ACTIONS => {
			"," => 94,
			")" => 206
		}
	},
	{#State 140
		ACTIONS => {
			":" => 207
		},
		DEFAULT => -98
	},
	{#State 141
		ACTIONS => {
			"," => 209,
			"]" => 208
		}
	},
	{#State 142
		DEFAULT => -97
	},
	{#State 143
		DEFAULT => -21
	},
	{#State 144
		DEFAULT => -139
	},
	{#State 145
		ACTIONS => {
			"," => 210,
			")" => 211
		}
	},
	{#State 146
		ACTIONS => {
			"," => 94,
			")" => 212
		}
	},
	{#State 147
		ACTIONS => {
			":" => 213
		}
	},
	{#State 148
		ACTIONS => {
			'AND_OP' => 88
		},
		DEFAULT => -176
	},
	{#State 149
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -159
	},
	{#State 150
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -157
	},
	{#State 151
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -160
	},
	{#State 152
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -158
	},
	{#State 153
		DEFAULT => -163
	},
	{#State 154
		DEFAULT => -165
	},
	{#State 155
		DEFAULT => -166
	},
	{#State 156
		DEFAULT => -164
	},
	{#State 157
		DEFAULT => -140
	},
	{#State 158
		DEFAULT => -174
	},
	{#State 159
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -155
	},
	{#State 160
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -151
	},
	{#State 161
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -153
	},
	{#State 162
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -152
	},
	{#State 163
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -154
	},
	{#State 164
		ACTIONS => {
			"=" => 75
		}
	},
	{#State 165
		DEFAULT => -19
	},
	{#State 166
		ACTIONS => {
			'IDENTIFIER' => 215
		},
		GOTOS => {
			'dim_args' => 214,
			'dim_arg' => 216
		}
	},
	{#State 167
		DEFAULT => -69
	},
	{#State 168
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 99,
			'input_declaration' => 217
		}
	},
	{#State 169
		DEFAULT => -84
	},
	{#State 170
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 101,
			'param_aux_declaration' => 218
		}
	},
	{#State 171
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 106,
			'obs_declaration' => 219
		}
	},
	{#State 172
		DEFAULT => -74
	},
	{#State 173
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 109,
			'state_aux_declaration' => 220
		}
	},
	{#State 174
		DEFAULT => -59
	},
	{#State 175
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 112,
			'noise_declaration' => 221
		}
	},
	{#State 176
		DEFAULT => -64
	},
	{#State 177
		DEFAULT => -108
	},
	{#State 178
		DEFAULT => -118
	},
	{#State 179
		ACTIONS => {
			"~" => 222,
			'SET_TO' => 223
		}
	},
	{#State 180
		DEFAULT => -114
	},
	{#State 181
		ACTIONS => {
			'NOISE' => 53,
			"}" => 177,
			'IDENTIFIER' => 183,
			'INPUT' => 40,
			"{" => 115,
			'CONST' => 57,
			'INLINE' => 56,
			'PARAM_AUX' => 43,
			'DIM' => 60,
			'STATE' => 61,
			'OBS' => 48,
			'DO' => 194,
			'PARAM' => 63,
			'STATE_AUX' => 52
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 178,
			'do' => 187,
			'block_start' => 113,
			'set_to' => 188,
			'state' => 44,
			'param_aux' => 45,
			'target' => 179,
			'var' => 189,
			'dim' => 180,
			'param' => 51,
			'inline' => 182,
			'spec' => 114,
			'block_definition' => 224,
			'dtarget' => 185,
			'const' => 186,
			'block_end' => 225,
			'action' => 191,
			'obs' => 62,
			'block' => 192,
			'varies_as' => 193,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 182
		DEFAULT => -113
	},
	{#State 183
		ACTIONS => {
			"(" => 6,
			"{" => -9,
			"/" => 227,
			"[" => 226
		},
		DEFAULT => -128
	},
	{#State 184
		DEFAULT => -110
	},
	{#State 185
		ACTIONS => {
			"=" => 228
		}
	},
	{#State 186
		DEFAULT => -112
	},
	{#State 187
		ACTIONS => {
			'THEN' => 229
		},
		DEFAULT => -111
	},
	{#State 188
		DEFAULT => -119
	},
	{#State 189
		DEFAULT => -115
	},
	{#State 190
		DEFAULT => -106
	},
	{#State 191
		DEFAULT => -116
	},
	{#State 192
		DEFAULT => -117
	},
	{#State 193
		DEFAULT => -120
	},
	{#State 194
		ACTIONS => {
			'IDENTIFIER' => 3,
			"{" => 115
		},
		GOTOS => {
			'spec' => 114,
			'block_start' => 113,
			'block' => 230
		}
	},
	{#State 195
		ACTIONS => {
			'NOISE' => 53,
			"}" => 177,
			'IDENTIFIER' => 183,
			'INPUT' => 40,
			"{" => 115,
			'CONST' => 57,
			'INLINE' => 56,
			'PARAM_AUX' => 43,
			'DIM' => 60,
			'STATE' => 61,
			'OBS' => 48,
			'DO' => 194,
			'PARAM' => 63,
			'STATE_AUX' => 52
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 178,
			'do' => 187,
			'block_start' => 113,
			'set_to' => 188,
			'state' => 44,
			'param_aux' => 45,
			'target' => 179,
			'var' => 189,
			'dim' => 180,
			'param' => 51,
			'block_definitions' => 231,
			'inline' => 182,
			'spec' => 114,
			'block_definition' => 184,
			'dtarget' => 185,
			'const' => 186,
			'block_end' => 232,
			'action' => 191,
			'obs' => 62,
			'block' => 192,
			'varies_as' => 193,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 196
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 233,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 197
		ACTIONS => {
			'IDENTIFIER' => 118
		},
		GOTOS => {
			'inline_declaration' => 234
		}
	},
	{#State 198
		ACTIONS => {
			'IDENTIFIER' => 122
		},
		GOTOS => {
			'const_declaration' => 235
		}
	},
	{#State 199
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 236,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 200
		ACTIONS => {
			'IDENTIFIER' => 3
		},
		GOTOS => {
			'spec' => 126,
			'dim_declaration' => 237
		}
	},
	{#State 201
		DEFAULT => -32
	},
	{#State 202
		DEFAULT => -54
	},
	{#State 203
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 129,
			'state_declaration' => 238
		}
	},
	{#State 204
		DEFAULT => -79
	},
	{#State 205
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 131,
			'param_declaration' => 239
		}
	},
	{#State 206
		DEFAULT => -5
	},
	{#State 207
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 240,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 208
		DEFAULT => -135
	},
	{#State 209
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 140,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'cast_expression' => 38,
			'index_arg' => 241
		}
	},
	{#State 210
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 24,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 138,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'named_args' => 242,
			'cast_expression' => 38
		}
	},
	{#State 211
		DEFAULT => -137
	},
	{#State 212
		DEFAULT => -138
	},
	{#State 213
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 243,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 214
		ACTIONS => {
			"," => 245,
			"]" => 244
		}
	},
	{#State 215
		DEFAULT => -88
	},
	{#State 216
		DEFAULT => -87
	},
	{#State 217
		DEFAULT => -67
	},
	{#State 218
		DEFAULT => -82
	},
	{#State 219
		DEFAULT => -72
	},
	{#State 220
		DEFAULT => -57
	},
	{#State 221
		DEFAULT => -62
	},
	{#State 222
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 246,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 223
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 247,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 224
		DEFAULT => -109
	},
	{#State 225
		DEFAULT => -105
	},
	{#State 226
		ACTIONS => {
			'IDENTIFIER' => 249,
			'INTEGER_LITERAL' => 250
		},
		GOTOS => {
			'dim_aliases' => 248,
			'dim_alias' => 251
		}
	},
	{#State 227
		ACTIONS => {
			'DT' => 252
		}
	},
	{#State 228
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 68,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 253,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 229
		ACTIONS => {
			'IDENTIFIER' => 3,
			"{" => 115
		},
		GOTOS => {
			'spec' => 114,
			'block_start' => 113,
			'block' => 254
		}
	},
	{#State 230
		DEFAULT => -101
	},
	{#State 231
		ACTIONS => {
			'NOISE' => 53,
			"}" => 177,
			'IDENTIFIER' => 183,
			'INPUT' => 40,
			"{" => 115,
			'CONST' => 57,
			'INLINE' => 56,
			'PARAM_AUX' => 43,
			'DIM' => 60,
			'STATE' => 61,
			'OBS' => 48,
			'DO' => 194,
			'PARAM' => 63,
			'STATE_AUX' => 52
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 178,
			'do' => 187,
			'block_start' => 113,
			'set_to' => 188,
			'state' => 44,
			'param_aux' => 45,
			'target' => 179,
			'var' => 189,
			'dim' => 180,
			'param' => 51,
			'inline' => 182,
			'spec' => 114,
			'block_definition' => 224,
			'dtarget' => 185,
			'const' => 186,
			'block_end' => 255,
			'action' => 191,
			'obs' => 62,
			'block' => 192,
			'varies_as' => 193,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 232
		DEFAULT => -104
	},
	{#State 233
		ACTIONS => {
			";" => 256
		},
		DEFAULT => -43
	},
	{#State 234
		DEFAULT => -40
	},
	{#State 235
		DEFAULT => -35
	},
	{#State 236
		ACTIONS => {
			";" => 257
		},
		DEFAULT => -38
	},
	{#State 237
		DEFAULT => -30
	},
	{#State 238
		DEFAULT => -52
	},
	{#State 239
		DEFAULT => -77
	},
	{#State 240
		DEFAULT => -99
	},
	{#State 241
		DEFAULT => -96
	},
	{#State 242
		ACTIONS => {
			"," => 94,
			")" => 258
		}
	},
	{#State 243
		DEFAULT => -178
	},
	{#State 244
		ACTIONS => {
			"(" => 259
		},
		DEFAULT => -15
	},
	{#State 245
		ACTIONS => {
			'IDENTIFIER' => 215
		},
		GOTOS => {
			'dim_arg' => 260
		}
	},
	{#State 246
		ACTIONS => {
			";" => 261
		},
		DEFAULT => -122
	},
	{#State 247
		ACTIONS => {
			";" => 262
		},
		DEFAULT => -124
	},
	{#State 248
		ACTIONS => {
			"," => 264,
			"]" => 263
		}
	},
	{#State 249
		ACTIONS => {
			"=" => 265
		},
		DEFAULT => -91
	},
	{#State 250
		ACTIONS => {
			":" => 266
		},
		DEFAULT => -94
	},
	{#State 251
		DEFAULT => -90
	},
	{#State 252
		DEFAULT => -130
	},
	{#State 253
		ACTIONS => {
			";" => 267
		},
		DEFAULT => -126
	},
	{#State 254
		DEFAULT => -102
	},
	{#State 255
		DEFAULT => -103
	},
	{#State 256
		DEFAULT => -42
	},
	{#State 257
		DEFAULT => -37
	},
	{#State 258
		DEFAULT => -136
	},
	{#State 259
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 24,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37,
			")" => 268
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'positional_args' => 269,
			'shift_expression' => 25,
			'positional_arg' => 28,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'named_args' => 270,
			'cast_expression' => 38
		}
	},
	{#State 260
		DEFAULT => -86
	},
	{#State 261
		DEFAULT => -121
	},
	{#State 262
		DEFAULT => -123
	},
	{#State 263
		ACTIONS => {
			"/" => 271
		},
		DEFAULT => -127
	},
	{#State 264
		ACTIONS => {
			'IDENTIFIER' => 249,
			'INTEGER_LITERAL' => 250
		},
		GOTOS => {
			'dim_alias' => 272
		}
	},
	{#State 265
		ACTIONS => {
			'INTEGER_LITERAL' => 273
		}
	},
	{#State 266
		ACTIONS => {
			'INTEGER_LITERAL' => 274
		}
	},
	{#State 267
		DEFAULT => -125
	},
	{#State 268
		DEFAULT => -14
	},
	{#State 269
		ACTIONS => {
			"," => 275,
			")" => 276
		}
	},
	{#State 270
		ACTIONS => {
			"," => 94,
			")" => 277
		}
	},
	{#State 271
		ACTIONS => {
			'DT' => 278
		}
	},
	{#State 272
		DEFAULT => -89
	},
	{#State 273
		ACTIONS => {
			":" => 279
		},
		DEFAULT => -92
	},
	{#State 274
		DEFAULT => -95
	},
	{#State 275
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 24,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37
		},
		GOTOS => {
			'conditional_expression' => 11,
			'and_expression' => 14,
			'inclusive_or_expression' => 15,
			'pow_expression' => 17,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 138,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'named_args' => 280,
			'cast_expression' => 38
		}
	},
	{#State 276
		DEFAULT => -12
	},
	{#State 277
		DEFAULT => -13
	},
	{#State 278
		DEFAULT => -129
	},
	{#State 279
		ACTIONS => {
			'INTEGER_LITERAL' => 281
		}
	},
	{#State 280
		ACTIONS => {
			"," => 94,
			")" => 282
		}
	},
	{#State 281
		DEFAULT => -93
	},
	{#State 282
		DEFAULT => -11
	}
],
                                  yyrules  =>
[
	[#Rule 0
		 '$start', 2, undef
	],
	[#Rule 1
		 'model', 5,
sub
#line 4 "share/bi.yp"
{ $_[0]->model($_[2]) }
	],
	[#Rule 2
		 'model', 4,
sub
#line 5 "share/bi.yp"
{ $_[0]->model($_[2]) }
	],
	[#Rule 3
		 'model_start', 1,
sub
#line 9 "share/bi.yp"
{ $_[0]->push_model }
	],
	[#Rule 4
		 'model_end', 1, undef
	],
	[#Rule 5
		 'spec', 6,
sub
#line 17 "share/bi.yp"
{ $_[0]->spec($_[1], [], $_[3], $_[5]) }
	],
	[#Rule 6
		 'spec', 4,
sub
#line 18 "share/bi.yp"
{ $_[0]->spec($_[1], [], $_[3]) }
	],
	[#Rule 7
		 'spec', 4,
sub
#line 19 "share/bi.yp"
{ $_[0]->spec($_[1], [], [], $_[3]) }
	],
	[#Rule 8
		 'spec', 3,
sub
#line 20 "share/bi.yp"
{ $_[0]->spec($_[1]) }
	],
	[#Rule 9
		 'spec', 1,
sub
#line 21 "share/bi.yp"
{ $_[0]->spec($_[1]) }
	],
	[#Rule 10
		 'array_spec', 1, undef
	],
	[#Rule 11
		 'array_spec', 9,
sub
#line 26 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3], $_[6], $_[8]) }
	],
	[#Rule 12
		 'array_spec', 7,
sub
#line 27 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3], $_[6]) }
	],
	[#Rule 13
		 'array_spec', 7,
sub
#line 28 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3], [], $_[6]) }
	],
	[#Rule 14
		 'array_spec', 6,
sub
#line 29 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3]) }
	],
	[#Rule 15
		 'array_spec', 4,
sub
#line 30 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3]) }
	],
	[#Rule 16
		 'positional_args', 3,
sub
#line 34 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 17
		 'positional_args', 1,
sub
#line 35 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 18
		 'positional_arg', 1,
sub
#line 39 "share/bi.yp"
{ $_[0]->positional_arg($_[1]) }
	],
	[#Rule 19
		 'named_args', 3,
sub
#line 43 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 20
		 'named_args', 1,
sub
#line 44 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 21
		 'named_arg', 3,
sub
#line 48 "share/bi.yp"
{ $_[0]->named_arg($_[1], $_[3]) }
	],
	[#Rule 22
		 'model_definitions', 2, undef
	],
	[#Rule 23
		 'model_definitions', 1, undef
	],
	[#Rule 24
		 'model_definition', 1, undef
	],
	[#Rule 25
		 'model_definition', 1, undef
	],
	[#Rule 26
		 'model_definition', 1, undef
	],
	[#Rule 27
		 'model_definition', 1, undef
	],
	[#Rule 28
		 'model_definition', 1, undef
	],
	[#Rule 29
		 'dim', 2, undef
	],
	[#Rule 30
		 'dim_declarations', 3, undef
	],
	[#Rule 31
		 'dim_declarations', 1, undef
	],
	[#Rule 32
		 'dim_declaration', 2,
sub
#line 74 "share/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 33
		 'dim_declaration', 1,
sub
#line 75 "share/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 34
		 'const', 2, undef
	],
	[#Rule 35
		 'const_declarations', 3, undef
	],
	[#Rule 36
		 'const_declarations', 1, undef
	],
	[#Rule 37
		 'const_declaration', 4,
sub
#line 88 "share/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 38
		 'const_declaration', 3,
sub
#line 89 "share/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 39
		 'inline', 2, undef
	],
	[#Rule 40
		 'inline_declarations', 3, undef
	],
	[#Rule 41
		 'inline_declarations', 1, undef
	],
	[#Rule 42
		 'inline_declaration', 4,
sub
#line 102 "share/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 43
		 'inline_declaration', 3,
sub
#line 103 "share/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 44
		 'var', 1, undef
	],
	[#Rule 45
		 'var', 1, undef
	],
	[#Rule 46
		 'var', 1, undef
	],
	[#Rule 47
		 'var', 1, undef
	],
	[#Rule 48
		 'var', 1, undef
	],
	[#Rule 49
		 'var', 1, undef
	],
	[#Rule 50
		 'var', 1, undef
	],
	[#Rule 51
		 'state', 2, undef
	],
	[#Rule 52
		 'state_declarations', 3, undef
	],
	[#Rule 53
		 'state_declarations', 1, undef
	],
	[#Rule 54
		 'state_declaration', 2,
sub
#line 126 "share/bi.yp"
{ $_[0]->var('state', $_[1]) }
	],
	[#Rule 55
		 'state_declaration', 1,
sub
#line 127 "share/bi.yp"
{ $_[0]->var('state', $_[1]) }
	],
	[#Rule 56
		 'state_aux', 2, undef
	],
	[#Rule 57
		 'state_aux_declarations', 3, undef
	],
	[#Rule 58
		 'state_aux_declarations', 1, undef
	],
	[#Rule 59
		 'state_aux_declaration', 2,
sub
#line 140 "share/bi.yp"
{ $_[0]->var('state_aux_', $_[1]) }
	],
	[#Rule 60
		 'state_aux_declaration', 1,
sub
#line 141 "share/bi.yp"
{ $_[0]->var('state_aux_', $_[1]) }
	],
	[#Rule 61
		 'noise', 2, undef
	],
	[#Rule 62
		 'noise_declarations', 3, undef
	],
	[#Rule 63
		 'noise_declarations', 1, undef
	],
	[#Rule 64
		 'noise_declaration', 2,
sub
#line 154 "share/bi.yp"
{ $_[0]->var('noise', $_[1]) }
	],
	[#Rule 65
		 'noise_declaration', 1,
sub
#line 155 "share/bi.yp"
{ $_[0]->var('noise', $_[1]) }
	],
	[#Rule 66
		 'input', 2, undef
	],
	[#Rule 67
		 'input_declarations', 3, undef
	],
	[#Rule 68
		 'input_declarations', 1, undef
	],
	[#Rule 69
		 'input_declaration', 2,
sub
#line 168 "share/bi.yp"
{ $_[0]->var('input', $_[1]) }
	],
	[#Rule 70
		 'input_declaration', 1,
sub
#line 169 "share/bi.yp"
{ $_[0]->var('input', $_[1]) }
	],
	[#Rule 71
		 'obs', 2, undef
	],
	[#Rule 72
		 'obs_declarations', 3, undef
	],
	[#Rule 73
		 'obs_declarations', 1, undef
	],
	[#Rule 74
		 'obs_declaration', 2,
sub
#line 182 "share/bi.yp"
{ $_[0]->var('obs', $_[1]) }
	],
	[#Rule 75
		 'obs_declaration', 1,
sub
#line 183 "share/bi.yp"
{ $_[0]->var('obs', $_[1]) }
	],
	[#Rule 76
		 'param', 2, undef
	],
	[#Rule 77
		 'param_declarations', 3, undef
	],
	[#Rule 78
		 'param_declarations', 1, undef
	],
	[#Rule 79
		 'param_declaration', 2,
sub
#line 196 "share/bi.yp"
{ $_[0]->var('param', $_[1]) }
	],
	[#Rule 80
		 'param_declaration', 1,
sub
#line 197 "share/bi.yp"
{ $_[0]->var('param', $_[1]) }
	],
	[#Rule 81
		 'param_aux', 2, undef
	],
	[#Rule 82
		 'param_aux_declarations', 3, undef
	],
	[#Rule 83
		 'param_aux_declarations', 1, undef
	],
	[#Rule 84
		 'param_aux_declaration', 2,
sub
#line 210 "share/bi.yp"
{ $_[0]->var('param_aux_', $_[1]) }
	],
	[#Rule 85
		 'param_aux_declaration', 1,
sub
#line 211 "share/bi.yp"
{ $_[0]->var('param_aux_', $_[1]) }
	],
	[#Rule 86
		 'dim_args', 3,
sub
#line 215 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 87
		 'dim_args', 1,
sub
#line 216 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 88
		 'dim_arg', 1,
sub
#line 220 "share/bi.yp"
{ $_[0]->dim_arg($_[1]) }
	],
	[#Rule 89
		 'dim_aliases', 3,
sub
#line 224 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 90
		 'dim_aliases', 1,
sub
#line 225 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 91
		 'dim_alias', 1,
sub
#line 229 "share/bi.yp"
{ $_[0]->dim_alias($_[1]) }
	],
	[#Rule 92
		 'dim_alias', 3,
sub
#line 230 "share/bi.yp"
{ $_[0]->dim_alias($_[1], $_[3]) }
	],
	[#Rule 93
		 'dim_alias', 5,
sub
#line 231 "share/bi.yp"
{ $_[0]->dim_alias($_[1], $_[3], $_[5]) }
	],
	[#Rule 94
		 'dim_alias', 1,
sub
#line 232 "share/bi.yp"
{ $_[0]->dim_alias(undef, $_[1]) }
	],
	[#Rule 95
		 'dim_alias', 3,
sub
#line 233 "share/bi.yp"
{ $_[0]->dim_alias(undef, $_[1], $_[3]) }
	],
	[#Rule 96
		 'index_args', 3,
sub
#line 237 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 97
		 'index_args', 1,
sub
#line 238 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 98
		 'index_arg', 1,
sub
#line 242 "share/bi.yp"
{ $_[0]->index($_[1]) }
	],
	[#Rule 99
		 'index_arg', 3,
sub
#line 243 "share/bi.yp"
{ $_[0]->range($_[1], $_[3]) }
	],
	[#Rule 100
		 'top_level', 2,
sub
#line 247 "share/bi.yp"
{ $_[0]->top_level($_[2]) }
	],
	[#Rule 101
		 'do', 2,
sub
#line 251 "share/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 102
		 'do', 3,
sub
#line 252 "share/bi.yp"
{ $_[0]->commit_block($_[3]) }
	],
	[#Rule 103
		 'block', 4,
sub
#line 256 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 104
		 'block', 3,
sub
#line 257 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 105
		 'block', 3,
sub
#line 258 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 106
		 'block', 2,
sub
#line 259 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 107
		 'block_start', 1,
sub
#line 263 "share/bi.yp"
{ $_[0]->push_block }
	],
	[#Rule 108
		 'block_end', 1, undef
	],
	[#Rule 109
		 'block_definitions', 2, undef
	],
	[#Rule 110
		 'block_definitions', 1, undef
	],
	[#Rule 111
		 'block_definition', 1, undef
	],
	[#Rule 112
		 'block_definition', 1, undef
	],
	[#Rule 113
		 'block_definition', 1, undef
	],
	[#Rule 114
		 'block_definition', 1, undef
	],
	[#Rule 115
		 'block_definition', 1, undef
	],
	[#Rule 116
		 'block_definition', 1, undef
	],
	[#Rule 117
		 'block_definition', 1, undef
	],
	[#Rule 118
		 'action', 1, undef
	],
	[#Rule 119
		 'action', 1, undef
	],
	[#Rule 120
		 'action', 1, undef
	],
	[#Rule 121
		 'distributed_as', 4,
sub
#line 292 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 122
		 'distributed_as', 3,
sub
#line 293 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 123
		 'set_to', 4,
sub
#line 297 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 124
		 'set_to', 3,
sub
#line 298 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 125
		 'varies_as', 4,
sub
#line 302 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 126
		 'varies_as', 3,
sub
#line 303 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 127
		 'target', 4,
sub
#line 307 "share/bi.yp"
{ $_[0]->target($_[1], $_[3]) }
	],
	[#Rule 128
		 'target', 1,
sub
#line 308 "share/bi.yp"
{ $_[0]->target($_[1]) }
	],
	[#Rule 129
		 'dtarget', 6,
sub
#line 312 "share/bi.yp"
{ $_[0]->dtarget($_[1], $_[3]) }
	],
	[#Rule 130
		 'dtarget', 3,
sub
#line 313 "share/bi.yp"
{ $_[0]->dtarget($_[1]) }
	],
	[#Rule 131
		 'postfix_expression', 1,
sub
#line 322 "share/bi.yp"
{ $_[0]->literal($_[1]) }
	],
	[#Rule 132
		 'postfix_expression', 1,
sub
#line 323 "share/bi.yp"
{ $_[0]->integer_literal($_[1]) }
	],
	[#Rule 133
		 'postfix_expression', 1,
sub
#line 324 "share/bi.yp"
{ $_[0]->string_literal($_[1]) }
	],
	[#Rule 134
		 'postfix_expression', 1,
sub
#line 325 "share/bi.yp"
{ $_[0]->identifier($_[1]) }
	],
	[#Rule 135
		 'postfix_expression', 4,
sub
#line 326 "share/bi.yp"
{ $_[0]->identifier($_[1], $_[3]) }
	],
	[#Rule 136
		 'postfix_expression', 6,
sub
#line 327 "share/bi.yp"
{ $_[0]->function($_[1], $_[3], $_[5]) }
	],
	[#Rule 137
		 'postfix_expression', 4,
sub
#line 328 "share/bi.yp"
{ $_[0]->function($_[1], $_[3]) }
	],
	[#Rule 138
		 'postfix_expression', 4,
sub
#line 329 "share/bi.yp"
{ $_[0]->function($_[1], undef, $_[3]) }
	],
	[#Rule 139
		 'postfix_expression', 3,
sub
#line 330 "share/bi.yp"
{ $_[0]->function($_[1]) }
	],
	[#Rule 140
		 'postfix_expression', 3,
sub
#line 331 "share/bi.yp"
{ $_[0]->parens($_[2]) }
	],
	[#Rule 141
		 'unary_expression', 1,
sub
#line 335 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 142
		 'unary_expression', 2,
sub
#line 336 "share/bi.yp"
{ $_[0]->unary_operator($_[1], $_[2]) }
	],
	[#Rule 143
		 'unary_operator', 1,
sub
#line 342 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 144
		 'unary_operator', 1,
sub
#line 343 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 145
		 'unary_operator', 1,
sub
#line 345 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 146
		 'cast_expression', 1,
sub
#line 349 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 147
		 'pow_expression', 1, undef
	],
	[#Rule 148
		 'pow_expression', 3,
sub
#line 355 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 149
		 'pow_expression', 3,
sub
#line 356 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 150
		 'multiplicative_expression', 1,
sub
#line 360 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 151
		 'multiplicative_expression', 3,
sub
#line 361 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 152
		 'multiplicative_expression', 3,
sub
#line 362 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 153
		 'multiplicative_expression', 3,
sub
#line 363 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 154
		 'multiplicative_expression', 3,
sub
#line 364 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 155
		 'multiplicative_expression', 3,
sub
#line 365 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 156
		 'additive_expression', 1,
sub
#line 369 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 157
		 'additive_expression', 3,
sub
#line 370 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 158
		 'additive_expression', 3,
sub
#line 371 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 159
		 'additive_expression', 3,
sub
#line 372 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 160
		 'additive_expression', 3,
sub
#line 373 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 161
		 'shift_expression', 1,
sub
#line 377 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 162
		 'relational_expression', 1,
sub
#line 383 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 163
		 'relational_expression', 3,
sub
#line 384 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 164
		 'relational_expression', 3,
sub
#line 385 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 165
		 'relational_expression', 3,
sub
#line 386 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 166
		 'relational_expression', 3,
sub
#line 387 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 167
		 'equality_expression', 1,
sub
#line 391 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 168
		 'equality_expression', 3,
sub
#line 392 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 169
		 'equality_expression', 3,
sub
#line 393 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 170
		 'and_expression', 1,
sub
#line 397 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 171
		 'exclusive_or_expression', 1,
sub
#line 402 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 172
		 'inclusive_or_expression', 1,
sub
#line 407 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 173
		 'logical_and_expression', 1,
sub
#line 412 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 174
		 'logical_and_expression', 3,
sub
#line 413 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 175
		 'logical_or_expression', 1,
sub
#line 417 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 176
		 'logical_or_expression', 3,
sub
#line 418 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 177
		 'conditional_expression', 1,
sub
#line 422 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 178
		 'conditional_expression', 5,
sub
#line 423 "share/bi.yp"
{ $_[0]->ternary_operator($_[1], $_[2], $_[3], $_[4], $_[5]) }
	],
	[#Rule 179
		 'expression', 1,
sub
#line 427 "share/bi.yp"
{ $_[0]->expression($_[1]) }
	]
],
                                  @_);
    bless($self,$class);
}

#line 430 "share/bi.yp"


1;

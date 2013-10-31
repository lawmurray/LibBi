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
		DEFAULT => -7
	},
	{#State 4
		ACTIONS => {
			"{" => 7
		},
		GOTOS => {
			'block_start' => 8
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
		DEFAULT => -105
	},
	{#State 8
		ACTIONS => {
			'NOISE' => 52,
			"}" => 39,
			'INPUT' => 40,
			'SUB' => 54,
			'CONST' => 56,
			'INLINE' => 55,
			'PARAM_AUX' => 42,
			'DIM' => 59,
			'STATE' => 60,
			'OBS' => 47,
			'PARAM' => 63,
			'STATE_AUX' => 51
		},
		GOTOS => {
			'inline' => 53,
			'input' => 41,
			'state' => 43,
			'model_definitions' => 57,
			'param_aux' => 44,
			'const' => 58,
			'top_level' => 45,
			'block_end' => 61,
			'var' => 46,
			'obs' => 62,
			'dim' => 48,
			'model_definition' => 49,
			'param' => 50,
			'state_aux' => 65,
			'noise' => 64
		}
	},
	{#State 9
		DEFAULT => -142
	},
	{#State 10
		DEFAULT => -141
	},
	{#State 11
		DEFAULT => -177
	},
	{#State 12
		DEFAULT => -130
	},
	{#State 13
		DEFAULT => -129
	},
	{#State 14
		DEFAULT => -169
	},
	{#State 15
		DEFAULT => -171
	},
	{#State 16
		DEFAULT => -143
	},
	{#State 17
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -148
	},
	{#State 18
		DEFAULT => -16
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
		DEFAULT => -144
	},
	{#State 21
		DEFAULT => -6
	},
	{#State 22
		ACTIONS => {
			'EQ_OP' => 70,
			'NE_OP' => 71
		},
		DEFAULT => -168
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
		DEFAULT => -132
	},
	{#State 25
		DEFAULT => -160
	},
	{#State 26
		ACTIONS => {
			"?" => 77,
			'OR_OP' => 78
		},
		DEFAULT => -175
	},
	{#State 27
		ACTIONS => {
			"-" => 79,
			'ELEM_ADD_OP' => 82,
			"+" => 80,
			'ELEM_SUB_OP' => 81
		},
		DEFAULT => -159
	},
	{#State 28
		DEFAULT => -15
	},
	{#State 29
		DEFAULT => -139
	},
	{#State 30
		DEFAULT => -18
	},
	{#State 31
		DEFAULT => -170
	},
	{#State 32
		ACTIONS => {
			"<" => 83,
			'LE_OP' => 84,
			'GE_OP' => 85,
			">" => 86
		},
		DEFAULT => -165
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
		DEFAULT => -173
	},
	{#State 35
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -154
	},
	{#State 36
		ACTIONS => {
			"," => 94,
			")" => 95
		}
	},
	{#State 37
		DEFAULT => -131
	},
	{#State 38
		DEFAULT => -145
	},
	{#State 39
		DEFAULT => -106
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
		DEFAULT => -45
	},
	{#State 42
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
	{#State 43
		DEFAULT => -42
	},
	{#State 44
		DEFAULT => -48
	},
	{#State 45
		DEFAULT => -26
	},
	{#State 46
		DEFAULT => -23
	},
	{#State 47
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
	{#State 48
		DEFAULT => -22
	},
	{#State 49
		DEFAULT => -21
	},
	{#State 50
		DEFAULT => -47
	},
	{#State 51
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
	{#State 52
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
	{#State 53
		DEFAULT => -25
	},
	{#State 54
		ACTIONS => {
			'IDENTIFIER' => 3,
			"{" => 7
		},
		GOTOS => {
			'spec' => 114,
			'block_start' => 113,
			'block' => 115
		}
	},
	{#State 55
		ACTIONS => {
			'IDENTIFIER' => 117
		},
		GOTOS => {
			'inline_declaration' => 116,
			'inline_declarations' => 118
		}
	},
	{#State 56
		ACTIONS => {
			'IDENTIFIER' => 121
		},
		GOTOS => {
			'const_declarations' => 119,
			'const_declaration' => 120
		}
	},
	{#State 57
		ACTIONS => {
			'NOISE' => 52,
			"}" => 39,
			'INPUT' => 40,
			'SUB' => 54,
			'CONST' => 56,
			'INLINE' => 55,
			'PARAM_AUX' => 42,
			'DIM' => 59,
			'STATE' => 60,
			'OBS' => 47,
			'PARAM' => 63,
			'STATE_AUX' => 51
		},
		GOTOS => {
			'inline' => 53,
			'input' => 41,
			'state' => 43,
			'param_aux' => 44,
			'const' => 58,
			'top_level' => 45,
			'block_end' => 123,
			'var' => 46,
			'obs' => 62,
			'dim' => 48,
			'model_definition' => 122,
			'param' => 50,
			'state_aux' => 65,
			'noise' => 64
		}
	},
	{#State 58
		DEFAULT => -24
	},
	{#State 59
		ACTIONS => {
			'IDENTIFIER' => 3
		},
		GOTOS => {
			'spec' => 125,
			'dim_declarations' => 124,
			'dim_declaration' => 126
		}
	},
	{#State 60
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'state_declarations' => 129,
			'array_spec' => 128,
			'state_declaration' => 127
		}
	},
	{#State 61
		DEFAULT => -2
	},
	{#State 62
		DEFAULT => -46
	},
	{#State 63
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 130,
			'param_declaration' => 131,
			'param_declarations' => 132
		}
	},
	{#State 64
		DEFAULT => -44
	},
	{#State 65
		DEFAULT => -43
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
			'cast_expression' => 133,
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
			'cast_expression' => 134,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 68
		ACTIONS => {
			"[" => 74,
			"(" => 76
		},
		DEFAULT => -132
	},
	{#State 69
		DEFAULT => -140
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
			'relational_expression' => 135,
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
			'relational_expression' => 136,
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
			'positional_arg' => 137,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'named_args' => 138,
			'cast_expression' => 38
		}
	},
	{#State 73
		DEFAULT => -4
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
			'expression' => 139,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'index_args' => 140,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'index_arg' => 141,
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
			'expression' => 142,
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
			")" => 143
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
			'positional_args' => 144,
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
			'named_args' => 145,
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
			'conditional_expression' => 146,
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
			'logical_and_expression' => 147,
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
			'multiplicative_expression' => 148,
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
			'multiplicative_expression' => 149,
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
			'multiplicative_expression' => 150,
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
			'multiplicative_expression' => 151,
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
			'shift_expression' => 152,
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
	{#State 87
		ACTIONS => {
			")" => 156
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
			'inclusive_or_expression' => 157,
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
			'pow_expression' => 158,
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
			'pow_expression' => 159,
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
			'pow_expression' => 160,
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
			'pow_expression' => 161,
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
			'pow_expression' => 162,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 94
		ACTIONS => {
			'IDENTIFIER' => 163
		},
		GOTOS => {
			'named_arg' => 164
		}
	},
	{#State 95
		DEFAULT => -5
	},
	{#State 96
		DEFAULT => -66
	},
	{#State 97
		DEFAULT => -8
	},
	{#State 98
		ACTIONS => {
			"[" => 165,
			"(" => 6
		},
		DEFAULT => -7
	},
	{#State 99
		ACTIONS => {
			";" => 166
		},
		DEFAULT => -68
	},
	{#State 100
		ACTIONS => {
			"," => 167
		},
		DEFAULT => -64
	},
	{#State 101
		ACTIONS => {
			";" => 168
		},
		DEFAULT => -83
	},
	{#State 102
		DEFAULT => -81
	},
	{#State 103
		ACTIONS => {
			"," => 169
		},
		DEFAULT => -79
	},
	{#State 104
		ACTIONS => {
			"," => 170
		},
		DEFAULT => -69
	},
	{#State 105
		DEFAULT => -71
	},
	{#State 106
		ACTIONS => {
			";" => 171
		},
		DEFAULT => -73
	},
	{#State 107
		ACTIONS => {
			"," => 172
		},
		DEFAULT => -54
	},
	{#State 108
		DEFAULT => -56
	},
	{#State 109
		ACTIONS => {
			";" => 173
		},
		DEFAULT => -58
	},
	{#State 110
		DEFAULT => -61
	},
	{#State 111
		ACTIONS => {
			"," => 174
		},
		DEFAULT => -59
	},
	{#State 112
		ACTIONS => {
			";" => 175
		},
		DEFAULT => -63
	},
	{#State 113
		ACTIONS => {
			'NOISE' => 52,
			"}" => 39,
			'IDENTIFIER' => 181,
			'INPUT' => 40,
			"{" => 7,
			'INLINE' => 55,
			'CONST' => 56,
			'PARAM_AUX' => 42,
			'DIM' => 59,
			'STATE' => 60,
			'OBS' => 47,
			'DO' => 192,
			'PARAM' => 63,
			'STATE_AUX' => 51
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 176,
			'block_start' => 113,
			'do' => 185,
			'set_to' => 186,
			'state' => 43,
			'param_aux' => 44,
			'target' => 177,
			'var' => 187,
			'dim' => 178,
			'param' => 50,
			'block_definitions' => 179,
			'inline' => 180,
			'spec' => 114,
			'block_definition' => 182,
			'dtarget' => 183,
			'const' => 184,
			'action' => 189,
			'block_end' => 188,
			'obs' => 62,
			'block' => 190,
			'varies_as' => 191,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 114
		ACTIONS => {
			"{" => 7
		},
		GOTOS => {
			'block_start' => 193
		}
	},
	{#State 115
		DEFAULT => -98
	},
	{#State 116
		DEFAULT => -39
	},
	{#State 117
		ACTIONS => {
			"=" => 194
		}
	},
	{#State 118
		ACTIONS => {
			"," => 195
		},
		DEFAULT => -37
	},
	{#State 119
		ACTIONS => {
			"," => 196
		},
		DEFAULT => -32
	},
	{#State 120
		DEFAULT => -34
	},
	{#State 121
		ACTIONS => {
			"=" => 197
		}
	},
	{#State 122
		DEFAULT => -20
	},
	{#State 123
		DEFAULT => -1
	},
	{#State 124
		ACTIONS => {
			"," => 198
		},
		DEFAULT => -27
	},
	{#State 125
		ACTIONS => {
			";" => 199
		},
		DEFAULT => -31
	},
	{#State 126
		DEFAULT => -29
	},
	{#State 127
		DEFAULT => -51
	},
	{#State 128
		ACTIONS => {
			";" => 200
		},
		DEFAULT => -53
	},
	{#State 129
		ACTIONS => {
			"," => 201
		},
		DEFAULT => -49
	},
	{#State 130
		ACTIONS => {
			";" => 202
		},
		DEFAULT => -78
	},
	{#State 131
		DEFAULT => -76
	},
	{#State 132
		ACTIONS => {
			"," => 203
		},
		DEFAULT => -74
	},
	{#State 133
		DEFAULT => -147
	},
	{#State 134
		DEFAULT => -146
	},
	{#State 135
		ACTIONS => {
			"<" => 83,
			'LE_OP' => 84,
			'GE_OP' => 85,
			">" => 86
		},
		DEFAULT => -166
	},
	{#State 136
		ACTIONS => {
			"<" => 83,
			'LE_OP' => 84,
			'GE_OP' => 85,
			">" => 86
		},
		DEFAULT => -167
	},
	{#State 137
		DEFAULT => -14
	},
	{#State 138
		ACTIONS => {
			"," => 94,
			")" => 204
		}
	},
	{#State 139
		ACTIONS => {
			":" => 205
		},
		DEFAULT => -96
	},
	{#State 140
		ACTIONS => {
			"," => 207,
			"]" => 206
		}
	},
	{#State 141
		DEFAULT => -95
	},
	{#State 142
		DEFAULT => -19
	},
	{#State 143
		DEFAULT => -137
	},
	{#State 144
		ACTIONS => {
			"," => 208,
			")" => 209
		}
	},
	{#State 145
		ACTIONS => {
			"," => 94,
			")" => 210
		}
	},
	{#State 146
		ACTIONS => {
			":" => 211
		}
	},
	{#State 147
		ACTIONS => {
			'AND_OP' => 88
		},
		DEFAULT => -174
	},
	{#State 148
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -157
	},
	{#State 149
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -155
	},
	{#State 150
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -158
	},
	{#State 151
		ACTIONS => {
			"%" => 89,
			"*" => 90,
			'ELEM_MUL_OP' => 92,
			"/" => 91,
			'ELEM_DIV_OP' => 93
		},
		DEFAULT => -156
	},
	{#State 152
		DEFAULT => -161
	},
	{#State 153
		DEFAULT => -163
	},
	{#State 154
		DEFAULT => -164
	},
	{#State 155
		DEFAULT => -162
	},
	{#State 156
		DEFAULT => -138
	},
	{#State 157
		DEFAULT => -172
	},
	{#State 158
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -153
	},
	{#State 159
		ACTIONS => {
			'ELEM_POW_OP' => 66,
			'POW_OP' => 67
		},
		DEFAULT => -149
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
		DEFAULT => -150
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
			"=" => 75
		}
	},
	{#State 164
		DEFAULT => -17
	},
	{#State 165
		ACTIONS => {
			'IDENTIFIER' => 213
		},
		GOTOS => {
			'dim_args' => 212,
			'dim_arg' => 214
		}
	},
	{#State 166
		DEFAULT => -67
	},
	{#State 167
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 99,
			'input_declaration' => 215
		}
	},
	{#State 168
		DEFAULT => -82
	},
	{#State 169
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 101,
			'param_aux_declaration' => 216
		}
	},
	{#State 170
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 106,
			'obs_declaration' => 217
		}
	},
	{#State 171
		DEFAULT => -72
	},
	{#State 172
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 109,
			'state_aux_declaration' => 218
		}
	},
	{#State 173
		DEFAULT => -57
	},
	{#State 174
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 112,
			'noise_declaration' => 219
		}
	},
	{#State 175
		DEFAULT => -62
	},
	{#State 176
		DEFAULT => -116
	},
	{#State 177
		ACTIONS => {
			"~" => 220,
			'SET_TO' => 221
		}
	},
	{#State 178
		DEFAULT => -112
	},
	{#State 179
		ACTIONS => {
			'NOISE' => 52,
			"}" => 39,
			'IDENTIFIER' => 181,
			'INPUT' => 40,
			"{" => 7,
			'CONST' => 56,
			'INLINE' => 55,
			'PARAM_AUX' => 42,
			'DIM' => 59,
			'STATE' => 60,
			'OBS' => 47,
			'DO' => 192,
			'PARAM' => 63,
			'STATE_AUX' => 51
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 176,
			'do' => 185,
			'block_start' => 113,
			'set_to' => 186,
			'state' => 43,
			'param_aux' => 44,
			'target' => 177,
			'var' => 187,
			'dim' => 178,
			'param' => 50,
			'inline' => 180,
			'spec' => 114,
			'block_definition' => 222,
			'dtarget' => 183,
			'const' => 184,
			'block_end' => 223,
			'action' => 189,
			'obs' => 62,
			'block' => 190,
			'varies_as' => 191,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 180
		DEFAULT => -111
	},
	{#State 181
		ACTIONS => {
			"(" => 6,
			"{" => -7,
			"/" => 225,
			"[" => 224
		},
		DEFAULT => -126
	},
	{#State 182
		DEFAULT => -108
	},
	{#State 183
		ACTIONS => {
			"=" => 226
		}
	},
	{#State 184
		DEFAULT => -110
	},
	{#State 185
		ACTIONS => {
			'THEN' => 227
		},
		DEFAULT => -109
	},
	{#State 186
		DEFAULT => -117
	},
	{#State 187
		DEFAULT => -113
	},
	{#State 188
		DEFAULT => -104
	},
	{#State 189
		DEFAULT => -114
	},
	{#State 190
		DEFAULT => -115
	},
	{#State 191
		DEFAULT => -118
	},
	{#State 192
		ACTIONS => {
			'IDENTIFIER' => 3,
			"{" => 7
		},
		GOTOS => {
			'spec' => 114,
			'block_start' => 113,
			'block' => 228
		}
	},
	{#State 193
		ACTIONS => {
			'NOISE' => 52,
			"}" => 39,
			'IDENTIFIER' => 181,
			'INPUT' => 40,
			"{" => 7,
			'CONST' => 56,
			'INLINE' => 55,
			'PARAM_AUX' => 42,
			'DIM' => 59,
			'STATE' => 60,
			'OBS' => 47,
			'DO' => 192,
			'PARAM' => 63,
			'STATE_AUX' => 51
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 176,
			'do' => 185,
			'block_start' => 113,
			'set_to' => 186,
			'state' => 43,
			'param_aux' => 44,
			'target' => 177,
			'var' => 187,
			'dim' => 178,
			'param' => 50,
			'block_definitions' => 229,
			'inline' => 180,
			'spec' => 114,
			'block_definition' => 182,
			'dtarget' => 183,
			'const' => 184,
			'block_end' => 230,
			'action' => 189,
			'obs' => 62,
			'block' => 190,
			'varies_as' => 191,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 194
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
			'expression' => 231,
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
	{#State 195
		ACTIONS => {
			'IDENTIFIER' => 117
		},
		GOTOS => {
			'inline_declaration' => 232
		}
	},
	{#State 196
		ACTIONS => {
			'IDENTIFIER' => 121
		},
		GOTOS => {
			'const_declaration' => 233
		}
	},
	{#State 197
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
			'expression' => 234,
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
	{#State 198
		ACTIONS => {
			'IDENTIFIER' => 3
		},
		GOTOS => {
			'spec' => 125,
			'dim_declaration' => 235
		}
	},
	{#State 199
		DEFAULT => -30
	},
	{#State 200
		DEFAULT => -52
	},
	{#State 201
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 128,
			'state_declaration' => 236
		}
	},
	{#State 202
		DEFAULT => -77
	},
	{#State 203
		ACTIONS => {
			'IDENTIFIER' => 98
		},
		GOTOS => {
			'spec' => 97,
			'array_spec' => 130,
			'param_declaration' => 237
		}
	},
	{#State 204
		DEFAULT => -3
	},
	{#State 205
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
			'expression' => 238,
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
	{#State 206
		DEFAULT => -133
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
			'expression' => 139,
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
			'index_arg' => 239
		}
	},
	{#State 208
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
			'positional_arg' => 137,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'named_args' => 240,
			'cast_expression' => 38
		}
	},
	{#State 209
		DEFAULT => -135
	},
	{#State 210
		DEFAULT => -136
	},
	{#State 211
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
			'conditional_expression' => 241,
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
	{#State 212
		ACTIONS => {
			"," => 243,
			"]" => 242
		}
	},
	{#State 213
		DEFAULT => -86
	},
	{#State 214
		DEFAULT => -85
	},
	{#State 215
		DEFAULT => -65
	},
	{#State 216
		DEFAULT => -80
	},
	{#State 217
		DEFAULT => -70
	},
	{#State 218
		DEFAULT => -55
	},
	{#State 219
		DEFAULT => -60
	},
	{#State 220
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
			'expression' => 244,
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
	{#State 221
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
			'expression' => 245,
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
	{#State 222
		DEFAULT => -107
	},
	{#State 223
		DEFAULT => -103
	},
	{#State 224
		ACTIONS => {
			'IDENTIFIER' => 247,
			'INTEGER_LITERAL' => 248
		},
		GOTOS => {
			'dim_aliases' => 246,
			'dim_alias' => 249
		}
	},
	{#State 225
		ACTIONS => {
			'DT' => 250
		}
	},
	{#State 226
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
			'expression' => 251,
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
	{#State 227
		ACTIONS => {
			'IDENTIFIER' => 3,
			"{" => 7
		},
		GOTOS => {
			'spec' => 114,
			'block_start' => 113,
			'block' => 252
		}
	},
	{#State 228
		DEFAULT => -99
	},
	{#State 229
		ACTIONS => {
			'NOISE' => 52,
			"}" => 39,
			'IDENTIFIER' => 181,
			'INPUT' => 40,
			"{" => 7,
			'CONST' => 56,
			'INLINE' => 55,
			'PARAM_AUX' => 42,
			'DIM' => 59,
			'STATE' => 60,
			'OBS' => 47,
			'DO' => 192,
			'PARAM' => 63,
			'STATE_AUX' => 51
		},
		GOTOS => {
			'input' => 41,
			'distributed_as' => 176,
			'do' => 185,
			'block_start' => 113,
			'set_to' => 186,
			'state' => 43,
			'param_aux' => 44,
			'target' => 177,
			'var' => 187,
			'dim' => 178,
			'param' => 50,
			'inline' => 180,
			'spec' => 114,
			'block_definition' => 222,
			'dtarget' => 183,
			'const' => 184,
			'block_end' => 253,
			'action' => 189,
			'obs' => 62,
			'block' => 190,
			'varies_as' => 191,
			'noise' => 64,
			'state_aux' => 65
		}
	},
	{#State 230
		DEFAULT => -102
	},
	{#State 231
		ACTIONS => {
			";" => 254
		},
		DEFAULT => -41
	},
	{#State 232
		DEFAULT => -38
	},
	{#State 233
		DEFAULT => -33
	},
	{#State 234
		ACTIONS => {
			";" => 255
		},
		DEFAULT => -36
	},
	{#State 235
		DEFAULT => -28
	},
	{#State 236
		DEFAULT => -50
	},
	{#State 237
		DEFAULT => -75
	},
	{#State 238
		DEFAULT => -97
	},
	{#State 239
		DEFAULT => -94
	},
	{#State 240
		ACTIONS => {
			"," => 94,
			")" => 256
		}
	},
	{#State 241
		DEFAULT => -176
	},
	{#State 242
		ACTIONS => {
			"(" => 257
		},
		DEFAULT => -13
	},
	{#State 243
		ACTIONS => {
			'IDENTIFIER' => 213
		},
		GOTOS => {
			'dim_arg' => 258
		}
	},
	{#State 244
		ACTIONS => {
			";" => 259
		},
		DEFAULT => -120
	},
	{#State 245
		ACTIONS => {
			";" => 260
		},
		DEFAULT => -122
	},
	{#State 246
		ACTIONS => {
			"," => 262,
			"]" => 261
		}
	},
	{#State 247
		ACTIONS => {
			"=" => 263
		},
		DEFAULT => -89
	},
	{#State 248
		ACTIONS => {
			":" => 264
		},
		DEFAULT => -92
	},
	{#State 249
		DEFAULT => -88
	},
	{#State 250
		DEFAULT => -128
	},
	{#State 251
		ACTIONS => {
			";" => 265
		},
		DEFAULT => -124
	},
	{#State 252
		DEFAULT => -100
	},
	{#State 253
		DEFAULT => -101
	},
	{#State 254
		DEFAULT => -40
	},
	{#State 255
		DEFAULT => -35
	},
	{#State 256
		DEFAULT => -134
	},
	{#State 257
		ACTIONS => {
			"-" => 9,
			'IDENTIFIER' => 24,
			"+" => 10,
			'INTEGER_LITERAL' => 12,
			'LITERAL' => 13,
			"!" => 16,
			"(" => 33,
			'STRING_LITERAL' => 37,
			")" => 266
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
			'positional_args' => 267,
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
			'named_args' => 268,
			'cast_expression' => 38
		}
	},
	{#State 258
		DEFAULT => -84
	},
	{#State 259
		DEFAULT => -119
	},
	{#State 260
		DEFAULT => -121
	},
	{#State 261
		ACTIONS => {
			"/" => 269
		},
		DEFAULT => -125
	},
	{#State 262
		ACTIONS => {
			'IDENTIFIER' => 247,
			'INTEGER_LITERAL' => 248
		},
		GOTOS => {
			'dim_alias' => 270
		}
	},
	{#State 263
		ACTIONS => {
			'INTEGER_LITERAL' => 271
		}
	},
	{#State 264
		ACTIONS => {
			'INTEGER_LITERAL' => 272
		}
	},
	{#State 265
		DEFAULT => -123
	},
	{#State 266
		DEFAULT => -12
	},
	{#State 267
		ACTIONS => {
			"," => 273,
			")" => 274
		}
	},
	{#State 268
		ACTIONS => {
			"," => 94,
			")" => 275
		}
	},
	{#State 269
		ACTIONS => {
			'DT' => 276
		}
	},
	{#State 270
		DEFAULT => -87
	},
	{#State 271
		ACTIONS => {
			":" => 277
		},
		DEFAULT => -90
	},
	{#State 272
		DEFAULT => -93
	},
	{#State 273
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
			'positional_arg' => 137,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'named_args' => 278,
			'cast_expression' => 38
		}
	},
	{#State 274
		DEFAULT => -10
	},
	{#State 275
		DEFAULT => -11
	},
	{#State 276
		DEFAULT => -127
	},
	{#State 277
		ACTIONS => {
			'INTEGER_LITERAL' => 279
		}
	},
	{#State 278
		ACTIONS => {
			"," => 94,
			")" => 280
		}
	},
	{#State 279
		DEFAULT => -91
	},
	{#State 280
		DEFAULT => -9
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
		 'spec', 6,
sub
#line 9 "share/bi.yp"
{ $_[0]->spec($_[1], [], $_[3], $_[5]) }
	],
	[#Rule 4
		 'spec', 4,
sub
#line 10 "share/bi.yp"
{ $_[0]->spec($_[1], [], $_[3]) }
	],
	[#Rule 5
		 'spec', 4,
sub
#line 11 "share/bi.yp"
{ $_[0]->spec($_[1], [], [], $_[3]) }
	],
	[#Rule 6
		 'spec', 3,
sub
#line 12 "share/bi.yp"
{ $_[0]->spec($_[1]) }
	],
	[#Rule 7
		 'spec', 1,
sub
#line 13 "share/bi.yp"
{ $_[0]->spec($_[1]) }
	],
	[#Rule 8
		 'array_spec', 1, undef
	],
	[#Rule 9
		 'array_spec', 9,
sub
#line 18 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3], $_[6], $_[8]) }
	],
	[#Rule 10
		 'array_spec', 7,
sub
#line 19 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3], $_[6]) }
	],
	[#Rule 11
		 'array_spec', 7,
sub
#line 20 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3], [], $_[6]) }
	],
	[#Rule 12
		 'array_spec', 6,
sub
#line 21 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3]) }
	],
	[#Rule 13
		 'array_spec', 4,
sub
#line 22 "share/bi.yp"
{ $_[0]->spec($_[1], $_[3]) }
	],
	[#Rule 14
		 'positional_args', 3,
sub
#line 26 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 15
		 'positional_args', 1,
sub
#line 27 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 16
		 'positional_arg', 1,
sub
#line 31 "share/bi.yp"
{ $_[0]->positional_arg($_[1]) }
	],
	[#Rule 17
		 'named_args', 3,
sub
#line 35 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 18
		 'named_args', 1,
sub
#line 36 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 19
		 'named_arg', 3,
sub
#line 40 "share/bi.yp"
{ $_[0]->named_arg($_[1], $_[3]) }
	],
	[#Rule 20
		 'model_definitions', 2, undef
	],
	[#Rule 21
		 'model_definitions', 1, undef
	],
	[#Rule 22
		 'model_definition', 1, undef
	],
	[#Rule 23
		 'model_definition', 1, undef
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
		 'dim', 2, undef
	],
	[#Rule 28
		 'dim_declarations', 3, undef
	],
	[#Rule 29
		 'dim_declarations', 1, undef
	],
	[#Rule 30
		 'dim_declaration', 2,
sub
#line 66 "share/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 31
		 'dim_declaration', 1,
sub
#line 67 "share/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 32
		 'const', 2, undef
	],
	[#Rule 33
		 'const_declarations', 3, undef
	],
	[#Rule 34
		 'const_declarations', 1, undef
	],
	[#Rule 35
		 'const_declaration', 4,
sub
#line 80 "share/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 36
		 'const_declaration', 3,
sub
#line 81 "share/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 37
		 'inline', 2, undef
	],
	[#Rule 38
		 'inline_declarations', 3, undef
	],
	[#Rule 39
		 'inline_declarations', 1, undef
	],
	[#Rule 40
		 'inline_declaration', 4,
sub
#line 94 "share/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 41
		 'inline_declaration', 3,
sub
#line 95 "share/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 42
		 'var', 1, undef
	],
	[#Rule 43
		 'var', 1, undef
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
		 'state', 2, undef
	],
	[#Rule 50
		 'state_declarations', 3, undef
	],
	[#Rule 51
		 'state_declarations', 1, undef
	],
	[#Rule 52
		 'state_declaration', 2,
sub
#line 118 "share/bi.yp"
{ $_[0]->var('state', $_[1]) }
	],
	[#Rule 53
		 'state_declaration', 1,
sub
#line 119 "share/bi.yp"
{ $_[0]->var('state', $_[1]) }
	],
	[#Rule 54
		 'state_aux', 2, undef
	],
	[#Rule 55
		 'state_aux_declarations', 3, undef
	],
	[#Rule 56
		 'state_aux_declarations', 1, undef
	],
	[#Rule 57
		 'state_aux_declaration', 2,
sub
#line 132 "share/bi.yp"
{ $_[0]->var('state_aux_', $_[1]) }
	],
	[#Rule 58
		 'state_aux_declaration', 1,
sub
#line 133 "share/bi.yp"
{ $_[0]->var('state_aux_', $_[1]) }
	],
	[#Rule 59
		 'noise', 2, undef
	],
	[#Rule 60
		 'noise_declarations', 3, undef
	],
	[#Rule 61
		 'noise_declarations', 1, undef
	],
	[#Rule 62
		 'noise_declaration', 2,
sub
#line 146 "share/bi.yp"
{ $_[0]->var('noise', $_[1]) }
	],
	[#Rule 63
		 'noise_declaration', 1,
sub
#line 147 "share/bi.yp"
{ $_[0]->var('noise', $_[1]) }
	],
	[#Rule 64
		 'input', 2, undef
	],
	[#Rule 65
		 'input_declarations', 3, undef
	],
	[#Rule 66
		 'input_declarations', 1, undef
	],
	[#Rule 67
		 'input_declaration', 2,
sub
#line 160 "share/bi.yp"
{ $_[0]->var('input', $_[1]) }
	],
	[#Rule 68
		 'input_declaration', 1,
sub
#line 161 "share/bi.yp"
{ $_[0]->var('input', $_[1]) }
	],
	[#Rule 69
		 'obs', 2, undef
	],
	[#Rule 70
		 'obs_declarations', 3, undef
	],
	[#Rule 71
		 'obs_declarations', 1, undef
	],
	[#Rule 72
		 'obs_declaration', 2,
sub
#line 174 "share/bi.yp"
{ $_[0]->var('obs', $_[1]) }
	],
	[#Rule 73
		 'obs_declaration', 1,
sub
#line 175 "share/bi.yp"
{ $_[0]->var('obs', $_[1]) }
	],
	[#Rule 74
		 'param', 2, undef
	],
	[#Rule 75
		 'param_declarations', 3, undef
	],
	[#Rule 76
		 'param_declarations', 1, undef
	],
	[#Rule 77
		 'param_declaration', 2,
sub
#line 188 "share/bi.yp"
{ $_[0]->var('param', $_[1]) }
	],
	[#Rule 78
		 'param_declaration', 1,
sub
#line 189 "share/bi.yp"
{ $_[0]->var('param', $_[1]) }
	],
	[#Rule 79
		 'param_aux', 2, undef
	],
	[#Rule 80
		 'param_aux_declarations', 3, undef
	],
	[#Rule 81
		 'param_aux_declarations', 1, undef
	],
	[#Rule 82
		 'param_aux_declaration', 2,
sub
#line 202 "share/bi.yp"
{ $_[0]->var('param_aux_', $_[1]) }
	],
	[#Rule 83
		 'param_aux_declaration', 1,
sub
#line 203 "share/bi.yp"
{ $_[0]->var('param_aux_', $_[1]) }
	],
	[#Rule 84
		 'dim_args', 3,
sub
#line 207 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 85
		 'dim_args', 1,
sub
#line 208 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 86
		 'dim_arg', 1,
sub
#line 212 "share/bi.yp"
{ $_[0]->dim_arg($_[1]) }
	],
	[#Rule 87
		 'dim_aliases', 3,
sub
#line 216 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 88
		 'dim_aliases', 1,
sub
#line 217 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 89
		 'dim_alias', 1,
sub
#line 221 "share/bi.yp"
{ $_[0]->dim_alias($_[1]) }
	],
	[#Rule 90
		 'dim_alias', 3,
sub
#line 222 "share/bi.yp"
{ $_[0]->dim_alias($_[1], $_[3]) }
	],
	[#Rule 91
		 'dim_alias', 5,
sub
#line 223 "share/bi.yp"
{ $_[0]->dim_alias($_[1], $_[3], $_[5]) }
	],
	[#Rule 92
		 'dim_alias', 1,
sub
#line 224 "share/bi.yp"
{ $_[0]->dim_alias(undef, $_[1]) }
	],
	[#Rule 93
		 'dim_alias', 3,
sub
#line 225 "share/bi.yp"
{ $_[0]->dim_alias(undef, $_[1], $_[3]) }
	],
	[#Rule 94
		 'index_args', 3,
sub
#line 229 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 95
		 'index_args', 1,
sub
#line 230 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 96
		 'index_arg', 1,
sub
#line 234 "share/bi.yp"
{ $_[0]->index($_[1]) }
	],
	[#Rule 97
		 'index_arg', 3,
sub
#line 235 "share/bi.yp"
{ $_[0]->range($_[1], $_[3]) }
	],
	[#Rule 98
		 'top_level', 2,
sub
#line 239 "share/bi.yp"
{ $_[0]->top_level($_[2]) }
	],
	[#Rule 99
		 'do', 2,
sub
#line 243 "share/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 100
		 'do', 3,
sub
#line 244 "share/bi.yp"
{ $_[0]->commit_block($_[3]) }
	],
	[#Rule 101
		 'block', 4,
sub
#line 248 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 102
		 'block', 3,
sub
#line 249 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 103
		 'block', 3,
sub
#line 250 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 104
		 'block', 2,
sub
#line 251 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 105
		 'block_start', 1,
sub
#line 255 "share/bi.yp"
{ $_[0]->push_block }
	],
	[#Rule 106
		 'block_end', 1, undef
	],
	[#Rule 107
		 'block_definitions', 2, undef
	],
	[#Rule 108
		 'block_definitions', 1, undef
	],
	[#Rule 109
		 'block_definition', 1, undef
	],
	[#Rule 110
		 'block_definition', 1, undef
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
		 'action', 1, undef
	],
	[#Rule 117
		 'action', 1, undef
	],
	[#Rule 118
		 'action', 1, undef
	],
	[#Rule 119
		 'distributed_as', 4,
sub
#line 284 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 120
		 'distributed_as', 3,
sub
#line 285 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 121
		 'set_to', 4,
sub
#line 289 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 122
		 'set_to', 3,
sub
#line 290 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 123
		 'varies_as', 4,
sub
#line 294 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 124
		 'varies_as', 3,
sub
#line 295 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 125
		 'target', 4,
sub
#line 299 "share/bi.yp"
{ $_[0]->target($_[1], $_[3]) }
	],
	[#Rule 126
		 'target', 1,
sub
#line 300 "share/bi.yp"
{ $_[0]->target($_[1]) }
	],
	[#Rule 127
		 'dtarget', 6,
sub
#line 304 "share/bi.yp"
{ $_[0]->dtarget($_[1], $_[3]) }
	],
	[#Rule 128
		 'dtarget', 3,
sub
#line 305 "share/bi.yp"
{ $_[0]->dtarget($_[1]) }
	],
	[#Rule 129
		 'postfix_expression', 1,
sub
#line 314 "share/bi.yp"
{ $_[0]->literal($_[1]) }
	],
	[#Rule 130
		 'postfix_expression', 1,
sub
#line 315 "share/bi.yp"
{ $_[0]->integer_literal($_[1]) }
	],
	[#Rule 131
		 'postfix_expression', 1,
sub
#line 316 "share/bi.yp"
{ $_[0]->string_literal($_[1]) }
	],
	[#Rule 132
		 'postfix_expression', 1,
sub
#line 317 "share/bi.yp"
{ $_[0]->identifier($_[1]) }
	],
	[#Rule 133
		 'postfix_expression', 4,
sub
#line 318 "share/bi.yp"
{ $_[0]->identifier($_[1], $_[3]) }
	],
	[#Rule 134
		 'postfix_expression', 6,
sub
#line 319 "share/bi.yp"
{ $_[0]->function($_[1], $_[3], $_[5]) }
	],
	[#Rule 135
		 'postfix_expression', 4,
sub
#line 320 "share/bi.yp"
{ $_[0]->function($_[1], $_[3]) }
	],
	[#Rule 136
		 'postfix_expression', 4,
sub
#line 321 "share/bi.yp"
{ $_[0]->function($_[1], undef, $_[3]) }
	],
	[#Rule 137
		 'postfix_expression', 3,
sub
#line 322 "share/bi.yp"
{ $_[0]->function($_[1]) }
	],
	[#Rule 138
		 'postfix_expression', 3,
sub
#line 323 "share/bi.yp"
{ $_[0]->parens($_[2]) }
	],
	[#Rule 139
		 'unary_expression', 1,
sub
#line 327 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 140
		 'unary_expression', 2,
sub
#line 328 "share/bi.yp"
{ $_[0]->unary_operator($_[1], $_[2]) }
	],
	[#Rule 141
		 'unary_operator', 1,
sub
#line 334 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 142
		 'unary_operator', 1,
sub
#line 335 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 143
		 'unary_operator', 1,
sub
#line 337 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 144
		 'cast_expression', 1,
sub
#line 341 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 145
		 'pow_expression', 1, undef
	],
	[#Rule 146
		 'pow_expression', 3,
sub
#line 347 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 147
		 'pow_expression', 3,
sub
#line 348 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 148
		 'multiplicative_expression', 1,
sub
#line 352 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 149
		 'multiplicative_expression', 3,
sub
#line 353 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 150
		 'multiplicative_expression', 3,
sub
#line 354 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 151
		 'multiplicative_expression', 3,
sub
#line 355 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 152
		 'multiplicative_expression', 3,
sub
#line 356 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 153
		 'multiplicative_expression', 3,
sub
#line 357 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 154
		 'additive_expression', 1,
sub
#line 361 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 155
		 'additive_expression', 3,
sub
#line 362 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 156
		 'additive_expression', 3,
sub
#line 363 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 157
		 'additive_expression', 3,
sub
#line 364 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 158
		 'additive_expression', 3,
sub
#line 365 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 159
		 'shift_expression', 1,
sub
#line 369 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 160
		 'relational_expression', 1,
sub
#line 375 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 161
		 'relational_expression', 3,
sub
#line 376 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 162
		 'relational_expression', 3,
sub
#line 377 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 163
		 'relational_expression', 3,
sub
#line 378 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 164
		 'relational_expression', 3,
sub
#line 379 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 165
		 'equality_expression', 1,
sub
#line 383 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 166
		 'equality_expression', 3,
sub
#line 384 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 167
		 'equality_expression', 3,
sub
#line 385 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 168
		 'and_expression', 1,
sub
#line 389 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 169
		 'exclusive_or_expression', 1,
sub
#line 394 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 170
		 'inclusive_or_expression', 1,
sub
#line 399 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 171
		 'logical_and_expression', 1,
sub
#line 404 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 172
		 'logical_and_expression', 3,
sub
#line 405 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 173
		 'logical_or_expression', 1,
sub
#line 409 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 174
		 'logical_or_expression', 3,
sub
#line 410 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 175
		 'conditional_expression', 1,
sub
#line 414 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 176
		 'conditional_expression', 5,
sub
#line 415 "share/bi.yp"
{ $_[0]->ternary_operator($_[1], $_[2], $_[3], $_[4], $_[5]) }
	],
	[#Rule 177
		 'expression', 1,
sub
#line 419 "share/bi.yp"
{ $_[0]->expression($_[1]) }
	]
],
                                  @_);
    bless($self,$class);
}

#line 422 "share/bi.yp"


1;

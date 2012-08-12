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
			'MODEL' => 3
		},
		GOTOS => {
			'top_levels' => 1,
			'top_level' => 2,
			'model' => 4
		}
	},
	{#State 1
		ACTIONS => {
			'' => 5,
			'MODEL' => 3
		},
		GOTOS => {
			'top_level' => 6,
			'model' => 4
		}
	},
	{#State 2
		DEFAULT => -1
	},
	{#State 3
		ACTIONS => {
			'IDENTIFIER' => 7
		},
		GOTOS => {
			'spec' => 8
		}
	},
	{#State 4
		DEFAULT => -3
	},
	{#State 5
		DEFAULT => 0
	},
	{#State 6
		DEFAULT => -2
	},
	{#State 7
		ACTIONS => {
			"(" => 9
		},
		DEFAULT => -10
	},
	{#State 8
		ACTIONS => {
			"{" => 10
		}
	},
	{#State 9
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 25,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38,
			")" => 22
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'positional_args' => 24,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'positional_arg' => 29,
			'postfix_expression' => 30,
			'named_arg' => 31,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'named_args' => 37,
			'cast_expression' => 39
		}
	},
	{#State 10
		ACTIONS => {
			'NOISE' => 51,
			"}" => 40,
			'INPUT' => 41,
			'SUB' => 53,
			'CONST' => 55,
			'INLINE' => 54,
			'PARAM_AUX' => 43,
			'DIM' => 58,
			'STATE' => 59,
			'OBS' => 46,
			'PARAM' => 62,
			'STATE_AUX' => 50
		},
		GOTOS => {
			'inline' => 52,
			'input' => 42,
			'state' => 44,
			'param_aux' => 45,
			'model_definitions' => 56,
			'const' => 57,
			'obs' => 61,
			'top_block' => 60,
			'dim' => 47,
			'model_definition' => 48,
			'param' => 49,
			'state_aux' => 64,
			'noise' => 63
		}
	},
	{#State 11
		DEFAULT => -127
	},
	{#State 12
		DEFAULT => -126
	},
	{#State 13
		DEFAULT => -158
	},
	{#State 14
		DEFAULT => -116
	},
	{#State 15
		DEFAULT => -115
	},
	{#State 16
		DEFAULT => -150
	},
	{#State 17
		DEFAULT => -152
	},
	{#State 18
		DEFAULT => -128
	},
	{#State 19
		DEFAULT => -18
	},
	{#State 20
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'unary_operator' => 20,
			'cast_expression' => 66,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 21
		DEFAULT => -129
	},
	{#State 22
		DEFAULT => -9
	},
	{#State 23
		ACTIONS => {
			'EQ_OP' => 67,
			'NE_OP' => 68
		},
		DEFAULT => -149
	},
	{#State 24
		ACTIONS => {
			"," => 69,
			")" => 70
		}
	},
	{#State 25
		ACTIONS => {
			"[" => 71,
			"=" => 72,
			"(" => 73
		},
		DEFAULT => -118
	},
	{#State 26
		DEFAULT => -141
	},
	{#State 27
		ACTIONS => {
			"?" => 74,
			'OR_OP' => 75
		},
		DEFAULT => -156
	},
	{#State 28
		ACTIONS => {
			"-" => 76,
			'ELEM_ADD_OP' => 79,
			"+" => 77,
			'ELEM_SUB_OP' => 78
		},
		DEFAULT => -140
	},
	{#State 29
		DEFAULT => -17
	},
	{#State 30
		DEFAULT => -124
	},
	{#State 31
		DEFAULT => -20
	},
	{#State 32
		DEFAULT => -151
	},
	{#State 33
		ACTIONS => {
			"<" => 80,
			'LE_OP' => 81,
			'GE_OP' => 82,
			">" => 83
		},
		DEFAULT => -146
	},
	{#State 34
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'equality_expression' => 23,
			'conditional_expression' => 84,
			'shift_expression' => 26,
			'postfix_expression' => 30,
			'additive_expression' => 28,
			'logical_or_expression' => 27,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21
		}
	},
	{#State 35
		ACTIONS => {
			'AND_OP' => 85
		},
		DEFAULT => -154
	},
	{#State 36
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -135
	},
	{#State 37
		ACTIONS => {
			"," => 90,
			")" => 91
		}
	},
	{#State 38
		DEFAULT => -117
	},
	{#State 39
		DEFAULT => -130
	},
	{#State 40
		DEFAULT => -5
	},
	{#State 41
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 93,
			'spec' => 92,
			'array_spec' => 95,
			'var_declarations' => 96
		}
	},
	{#State 42
		DEFAULT => -28
	},
	{#State 43
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 93,
			'spec' => 92,
			'array_spec' => 95,
			'var_declarations' => 97
		}
	},
	{#State 44
		DEFAULT => -25
	},
	{#State 45
		DEFAULT => -31
	},
	{#State 46
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 93,
			'spec' => 92,
			'array_spec' => 95,
			'var_declarations' => 98
		}
	},
	{#State 47
		DEFAULT => -24
	},
	{#State 48
		DEFAULT => -23
	},
	{#State 49
		DEFAULT => -30
	},
	{#State 50
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 93,
			'spec' => 92,
			'array_spec' => 95,
			'var_declarations' => 99
		}
	},
	{#State 51
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 93,
			'spec' => 92,
			'array_spec' => 95,
			'var_declarations' => 100
		}
	},
	{#State 52
		DEFAULT => -33
	},
	{#State 53
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 102
		},
		GOTOS => {
			'spec' => 101,
			'block' => 103
		}
	},
	{#State 54
		ACTIONS => {
			'IDENTIFIER' => 105
		},
		GOTOS => {
			'inline_declaration' => 104,
			'inline_declarations' => 106
		}
	},
	{#State 55
		ACTIONS => {
			'IDENTIFIER' => 109
		},
		GOTOS => {
			'const_declarations' => 107,
			'const_declaration' => 108
		}
	},
	{#State 56
		ACTIONS => {
			'NOISE' => 51,
			"}" => 110,
			'INPUT' => 41,
			'SUB' => 53,
			'CONST' => 55,
			'INLINE' => 54,
			'PARAM_AUX' => 43,
			'DIM' => 58,
			'STATE' => 59,
			'OBS' => 46,
			'PARAM' => 62,
			'STATE_AUX' => 50
		},
		GOTOS => {
			'inline' => 52,
			'input' => 42,
			'state' => 44,
			'param_aux' => 45,
			'const' => 57,
			'obs' => 61,
			'top_block' => 60,
			'dim' => 47,
			'model_definition' => 111,
			'param' => 49,
			'state_aux' => 64,
			'noise' => 63
		}
	},
	{#State 57
		DEFAULT => -32
	},
	{#State 58
		ACTIONS => {
			'IDENTIFIER' => 7
		},
		GOTOS => {
			'spec' => 113,
			'dim_declarations' => 112,
			'dim_declaration' => 114
		}
	},
	{#State 59
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 93,
			'spec' => 92,
			'array_spec' => 95,
			'var_declarations' => 115
		}
	},
	{#State 60
		DEFAULT => -34
	},
	{#State 61
		DEFAULT => -29
	},
	{#State 62
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 93,
			'spec' => 92,
			'array_spec' => 95,
			'var_declarations' => 116
		}
	},
	{#State 63
		DEFAULT => -27
	},
	{#State 64
		DEFAULT => -26
	},
	{#State 65
		ACTIONS => {
			"[" => 71,
			"(" => 73
		},
		DEFAULT => -118
	},
	{#State 66
		DEFAULT => -125
	},
	{#State 67
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'shift_expression' => 26,
			'postfix_expression' => 30,
			'additive_expression' => 28,
			'relational_expression' => 117,
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'cast_expression' => 39
		}
	},
	{#State 68
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'shift_expression' => 26,
			'postfix_expression' => 30,
			'additive_expression' => 28,
			'relational_expression' => 118,
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'cast_expression' => 39
		}
	},
	{#State 69
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 25,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'positional_arg' => 119,
			'named_arg' => 31,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'named_args' => 120,
			'cast_expression' => 39
		}
	},
	{#State 70
		DEFAULT => -7
	},
	{#State 71
		ACTIONS => {
			"-" => 121,
			'IDENTIFIER' => 125,
			'INTEGER_LITERAL' => 122
		},
		GOTOS => {
			'range_arg' => 124,
			'range_args' => 127,
			'offset_arg' => 126,
			'offset_args' => 123
		}
	},
	{#State 72
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 128,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 73
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38,
			")" => 129
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'positional_args' => 130,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'positional_arg' => 29,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'cast_expression' => 39
		}
	},
	{#State 74
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'equality_expression' => 23,
			'conditional_expression' => 131,
			'shift_expression' => 26,
			'postfix_expression' => 30,
			'additive_expression' => 28,
			'logical_or_expression' => 27,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21
		}
	},
	{#State 75
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'equality_expression' => 23,
			'shift_expression' => 26,
			'postfix_expression' => 30,
			'additive_expression' => 28,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 132,
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21
		}
	},
	{#State 76
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 133,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 77
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 134,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 78
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 135,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 79
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 136,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 80
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'shift_expression' => 137,
			'unary_expression' => 21,
			'postfix_expression' => 30,
			'additive_expression' => 28
		}
	},
	{#State 81
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'shift_expression' => 138,
			'unary_expression' => 21,
			'postfix_expression' => 30,
			'additive_expression' => 28
		}
	},
	{#State 82
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'shift_expression' => 139,
			'unary_expression' => 21,
			'postfix_expression' => 30,
			'additive_expression' => 28
		}
	},
	{#State 83
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'shift_expression' => 140,
			'unary_expression' => 21,
			'postfix_expression' => 30,
			'additive_expression' => 28
		}
	},
	{#State 84
		ACTIONS => {
			")" => 141
		}
	},
	{#State 85
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'equality_expression' => 23,
			'shift_expression' => 26,
			'postfix_expression' => 30,
			'additive_expression' => 28,
			'and_expression' => 16,
			'inclusive_or_expression' => 142,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21
		}
	},
	{#State 86
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'unary_operator' => 20,
			'cast_expression' => 143,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 87
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'unary_operator' => 20,
			'cast_expression' => 144,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 88
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'unary_operator' => 20,
			'cast_expression' => 145,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 89
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'unary_operator' => 20,
			'cast_expression' => 146,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 90
		ACTIONS => {
			'IDENTIFIER' => 147
		},
		GOTOS => {
			'named_arg' => 148
		}
	},
	{#State 91
		DEFAULT => -8
	},
	{#State 92
		DEFAULT => -67
	},
	{#State 93
		DEFAULT => -65
	},
	{#State 94
		ACTIONS => {
			"(" => 9,
			"[" => 149
		},
		DEFAULT => -10
	},
	{#State 95
		DEFAULT => -66
	},
	{#State 96
		ACTIONS => {
			";" => 150,
			"," => 151
		},
		DEFAULT => -57
	},
	{#State 97
		ACTIONS => {
			";" => 152,
			"," => 151
		},
		DEFAULT => -63
	},
	{#State 98
		ACTIONS => {
			";" => 153,
			"," => 151
		},
		DEFAULT => -59
	},
	{#State 99
		ACTIONS => {
			";" => 154,
			"," => 151
		},
		DEFAULT => -53
	},
	{#State 100
		ACTIONS => {
			";" => 155,
			"," => 151
		},
		DEFAULT => -55
	},
	{#State 101
		ACTIONS => {
			"{" => 156
		}
	},
	{#State 102
		ACTIONS => {
			"}" => 157,
			'IDENTIFIER' => 163,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 167
		},
		GOTOS => {
			'inline' => 162,
			'spec' => 101,
			'distributed_as' => 158,
			'do' => 159,
			'block_definition' => 164,
			'set_to' => 160,
			'const' => 165,
			'block' => 166,
			'block_definitions' => 161
		}
	},
	{#State 103
		DEFAULT => -88
	},
	{#State 104
		DEFAULT => -47
	},
	{#State 105
		ACTIONS => {
			"=" => 168
		}
	},
	{#State 106
		ACTIONS => {
			"," => 169
		},
		DEFAULT => -45
	},
	{#State 107
		ACTIONS => {
			"," => 170
		},
		DEFAULT => -40
	},
	{#State 108
		DEFAULT => -42
	},
	{#State 109
		ACTIONS => {
			"=" => 171
		}
	},
	{#State 110
		DEFAULT => -4
	},
	{#State 111
		DEFAULT => -22
	},
	{#State 112
		ACTIONS => {
			"," => 172
		},
		DEFAULT => -35
	},
	{#State 113
		ACTIONS => {
			";" => 173
		},
		DEFAULT => -39
	},
	{#State 114
		DEFAULT => -37
	},
	{#State 115
		ACTIONS => {
			";" => 174,
			"," => 151
		},
		DEFAULT => -51
	},
	{#State 116
		ACTIONS => {
			";" => 175,
			"," => 151
		},
		DEFAULT => -61
	},
	{#State 117
		ACTIONS => {
			"<" => 80,
			'LE_OP' => 81,
			'GE_OP' => 82,
			">" => 83
		},
		DEFAULT => -147
	},
	{#State 118
		ACTIONS => {
			"<" => 80,
			'LE_OP' => 81,
			'GE_OP' => 82,
			">" => 83
		},
		DEFAULT => -148
	},
	{#State 119
		DEFAULT => -16
	},
	{#State 120
		ACTIONS => {
			"," => 90,
			")" => 176
		}
	},
	{#State 121
		ACTIONS => {
			'INTEGER_LITERAL' => 177
		}
	},
	{#State 122
		ACTIONS => {
			":" => 178
		},
		DEFAULT => -84
	},
	{#State 123
		ACTIONS => {
			"," => 179,
			"]" => 180
		}
	},
	{#State 124
		DEFAULT => -86
	},
	{#State 125
		ACTIONS => {
			"-" => 181,
			"+" => 182
		},
		DEFAULT => -80
	},
	{#State 126
		DEFAULT => -79
	},
	{#State 127
		ACTIONS => {
			"," => 183,
			"]" => 184
		}
	},
	{#State 128
		DEFAULT => -21
	},
	{#State 129
		DEFAULT => -122
	},
	{#State 130
		ACTIONS => {
			"," => 185,
			")" => 186
		}
	},
	{#State 131
		ACTIONS => {
			":" => 187
		}
	},
	{#State 132
		ACTIONS => {
			'AND_OP' => 85
		},
		DEFAULT => -155
	},
	{#State 133
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -138
	},
	{#State 134
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -136
	},
	{#State 135
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -139
	},
	{#State 136
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -137
	},
	{#State 137
		DEFAULT => -142
	},
	{#State 138
		DEFAULT => -144
	},
	{#State 139
		DEFAULT => -145
	},
	{#State 140
		DEFAULT => -143
	},
	{#State 141
		DEFAULT => -123
	},
	{#State 142
		DEFAULT => -153
	},
	{#State 143
		DEFAULT => -131
	},
	{#State 144
		DEFAULT => -133
	},
	{#State 145
		DEFAULT => -132
	},
	{#State 146
		DEFAULT => -134
	},
	{#State 147
		ACTIONS => {
			"=" => 72
		}
	},
	{#State 148
		DEFAULT => -19
	},
	{#State 149
		ACTIONS => {
			'IDENTIFIER' => 190
		},
		GOTOS => {
			'dim_args' => 188,
			'dim_arg' => 189
		}
	},
	{#State 150
		DEFAULT => -56
	},
	{#State 151
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 191,
			'spec' => 92,
			'array_spec' => 95
		}
	},
	{#State 152
		DEFAULT => -62
	},
	{#State 153
		DEFAULT => -58
	},
	{#State 154
		DEFAULT => -52
	},
	{#State 155
		DEFAULT => -54
	},
	{#State 156
		ACTIONS => {
			"}" => 192,
			'IDENTIFIER' => 163,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 167
		},
		GOTOS => {
			'inline' => 162,
			'spec' => 101,
			'distributed_as' => 158,
			'do' => 159,
			'block_definition' => 164,
			'set_to' => 160,
			'const' => 165,
			'block' => 166,
			'block_definitions' => 193
		}
	},
	{#State 157
		DEFAULT => -94
	},
	{#State 158
		DEFAULT => -98
	},
	{#State 159
		ACTIONS => {
			'THEN' => 194
		},
		DEFAULT => -97
	},
	{#State 160
		DEFAULT => -99
	},
	{#State 161
		ACTIONS => {
			"}" => 195,
			'IDENTIFIER' => 163,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 167
		},
		GOTOS => {
			'inline' => 162,
			'spec' => 101,
			'const' => 165,
			'distributed_as' => 158,
			'do' => 159,
			'block' => 166,
			'block_definition' => 196,
			'set_to' => 160
		}
	},
	{#State 162
		DEFAULT => -101
	},
	{#State 163
		ACTIONS => {
			"(" => 9,
			"~" => 197,
			"[" => 198,
			'SET_TO' => 199
		},
		DEFAULT => -10
	},
	{#State 164
		DEFAULT => -96
	},
	{#State 165
		DEFAULT => -100
	},
	{#State 166
		DEFAULT => -102
	},
	{#State 167
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 102
		},
		GOTOS => {
			'spec' => 101,
			'block' => 200
		}
	},
	{#State 168
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 201,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 169
		ACTIONS => {
			'IDENTIFIER' => 105
		},
		GOTOS => {
			'inline_declaration' => 202
		}
	},
	{#State 170
		ACTIONS => {
			'IDENTIFIER' => 109
		},
		GOTOS => {
			'const_declaration' => 203
		}
	},
	{#State 171
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 204,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 172
		ACTIONS => {
			'IDENTIFIER' => 7
		},
		GOTOS => {
			'spec' => 113,
			'dim_declaration' => 205
		}
	},
	{#State 173
		DEFAULT => -38
	},
	{#State 174
		DEFAULT => -50
	},
	{#State 175
		DEFAULT => -60
	},
	{#State 176
		DEFAULT => -6
	},
	{#State 177
		DEFAULT => -83
	},
	{#State 178
		ACTIONS => {
			'INTEGER_LITERAL' => 206
		}
	},
	{#State 179
		ACTIONS => {
			"-" => 121,
			'IDENTIFIER' => 125,
			'INTEGER_LITERAL' => 207
		},
		GOTOS => {
			'offset_arg' => 208
		}
	},
	{#State 180
		DEFAULT => -119
	},
	{#State 181
		ACTIONS => {
			'INTEGER_LITERAL' => 209
		}
	},
	{#State 182
		ACTIONS => {
			'INTEGER_LITERAL' => 210
		}
	},
	{#State 183
		ACTIONS => {
			'INTEGER_LITERAL' => 211
		},
		GOTOS => {
			'range_arg' => 212
		}
	},
	{#State 184
		DEFAULT => -120
	},
	{#State 185
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'positional_arg' => 119,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 186
		DEFAULT => -121
	},
	{#State 187
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'equality_expression' => 23,
			'conditional_expression' => 213,
			'shift_expression' => 26,
			'postfix_expression' => 30,
			'additive_expression' => 28,
			'logical_or_expression' => 27,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'unary_operator' => 20,
			'cast_expression' => 39,
			'unary_expression' => 21
		}
	},
	{#State 188
		ACTIONS => {
			"," => 214,
			"]" => 215
		}
	},
	{#State 189
		DEFAULT => -69
	},
	{#State 190
		DEFAULT => -70
	},
	{#State 191
		DEFAULT => -64
	},
	{#State 192
		DEFAULT => -92
	},
	{#State 193
		ACTIONS => {
			"}" => 216,
			'IDENTIFIER' => 163,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 167
		},
		GOTOS => {
			'inline' => 162,
			'spec' => 101,
			'const' => 165,
			'distributed_as' => 158,
			'do' => 159,
			'block' => 166,
			'block_definition' => 196,
			'set_to' => 160
		}
	},
	{#State 194
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 102
		},
		GOTOS => {
			'spec' => 101,
			'block' => 217
		}
	},
	{#State 195
		DEFAULT => -93
	},
	{#State 196
		DEFAULT => -95
	},
	{#State 197
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 218,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 198
		ACTIONS => {
			'IDENTIFIER' => 222,
			'INTEGER_LITERAL' => 219
		},
		GOTOS => {
			'dim_aliases' => 220,
			'dim_alias' => 224,
			'dim_range' => 223,
			'dim_ranges' => 221
		}
	},
	{#State 199
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 225,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 200
		DEFAULT => -89
	},
	{#State 201
		ACTIONS => {
			";" => 226
		},
		DEFAULT => -49
	},
	{#State 202
		DEFAULT => -46
	},
	{#State 203
		DEFAULT => -41
	},
	{#State 204
		ACTIONS => {
			";" => 227
		},
		DEFAULT => -44
	},
	{#State 205
		DEFAULT => -36
	},
	{#State 206
		DEFAULT => -87
	},
	{#State 207
		DEFAULT => -84
	},
	{#State 208
		DEFAULT => -78
	},
	{#State 209
		DEFAULT => -81
	},
	{#State 210
		DEFAULT => -82
	},
	{#State 211
		ACTIONS => {
			":" => 178
		}
	},
	{#State 212
		DEFAULT => -85
	},
	{#State 213
		DEFAULT => -157
	},
	{#State 214
		ACTIONS => {
			'IDENTIFIER' => 190
		},
		GOTOS => {
			'dim_arg' => 228
		}
	},
	{#State 215
		ACTIONS => {
			"(" => 229
		},
		DEFAULT => -15
	},
	{#State 216
		DEFAULT => -91
	},
	{#State 217
		DEFAULT => -90
	},
	{#State 218
		ACTIONS => {
			";" => 230
		},
		DEFAULT => -108
	},
	{#State 219
		ACTIONS => {
			":" => 231
		},
		DEFAULT => -77
	},
	{#State 220
		ACTIONS => {
			"," => 232,
			"]" => 233
		}
	},
	{#State 221
		ACTIONS => {
			"," => 234,
			"]" => 235
		}
	},
	{#State 222
		DEFAULT => -73
	},
	{#State 223
		DEFAULT => -75
	},
	{#State 224
		DEFAULT => -72
	},
	{#State 225
		ACTIONS => {
			";" => 236
		},
		DEFAULT => -114
	},
	{#State 226
		DEFAULT => -48
	},
	{#State 227
		DEFAULT => -43
	},
	{#State 228
		DEFAULT => -68
	},
	{#State 229
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 25,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38,
			")" => 237
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'positional_args' => 238,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'positional_arg' => 29,
			'named_arg' => 31,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'named_args' => 239,
			'cast_expression' => 39
		}
	},
	{#State 230
		DEFAULT => -107
	},
	{#State 231
		ACTIONS => {
			'INTEGER_LITERAL' => 240
		}
	},
	{#State 232
		ACTIONS => {
			'IDENTIFIER' => 222
		},
		GOTOS => {
			'dim_alias' => 241
		}
	},
	{#State 233
		ACTIONS => {
			"~" => 242,
			'SET_TO' => 243
		}
	},
	{#State 234
		ACTIONS => {
			'INTEGER_LITERAL' => 219
		},
		GOTOS => {
			'dim_range' => 244
		}
	},
	{#State 235
		ACTIONS => {
			"~" => 245,
			'SET_TO' => 246
		}
	},
	{#State 236
		DEFAULT => -113
	},
	{#State 237
		DEFAULT => -14
	},
	{#State 238
		ACTIONS => {
			"," => 247,
			")" => 248
		}
	},
	{#State 239
		ACTIONS => {
			"," => 90,
			")" => 249
		}
	},
	{#State 240
		DEFAULT => -76
	},
	{#State 241
		DEFAULT => -71
	},
	{#State 242
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 250,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 243
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 251,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 244
		DEFAULT => -74
	},
	{#State 245
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 252,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 246
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 65,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 253,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'logical_and_expression' => 35,
			'multiplicative_expression' => 36,
			'cast_expression' => 39
		}
	},
	{#State 247
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 25,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'shift_expression' => 26,
			'logical_or_expression' => 27,
			'additive_expression' => 28,
			'postfix_expression' => 30,
			'positional_arg' => 119,
			'named_arg' => 31,
			'exclusive_or_expression' => 32,
			'relational_expression' => 33,
			'multiplicative_expression' => 36,
			'logical_and_expression' => 35,
			'named_args' => 254,
			'cast_expression' => 39
		}
	},
	{#State 248
		DEFAULT => -12
	},
	{#State 249
		DEFAULT => -13
	},
	{#State 250
		ACTIONS => {
			";" => 255
		},
		DEFAULT => -104
	},
	{#State 251
		ACTIONS => {
			";" => 256
		},
		DEFAULT => -110
	},
	{#State 252
		ACTIONS => {
			";" => 257
		},
		DEFAULT => -106
	},
	{#State 253
		ACTIONS => {
			";" => 258
		},
		DEFAULT => -112
	},
	{#State 254
		ACTIONS => {
			"," => 90,
			")" => 259
		}
	},
	{#State 255
		DEFAULT => -103
	},
	{#State 256
		DEFAULT => -109
	},
	{#State 257
		DEFAULT => -105
	},
	{#State 258
		DEFAULT => -111
	},
	{#State 259
		DEFAULT => -11
	}
],
                                  yyrules  =>
[
	[#Rule 0
		 '$start', 2, undef
	],
	[#Rule 1
		 'top_levels', 1, undef
	],
	[#Rule 2
		 'top_levels', 2, undef
	],
	[#Rule 3
		 'top_level', 1, undef
	],
	[#Rule 4
		 'model', 5,
sub
#line 13 "share/bi.yp"
{ $_[0]->model($_[2], $_[4]) }
	],
	[#Rule 5
		 'model', 4,
sub
#line 14 "share/bi.yp"
{ $_[0]->model($_[2]) }
	],
	[#Rule 6
		 'spec', 6,
sub
#line 18 "share/bi.yp"
{ $_[0]->spec($_[1], [], $_[3], $_[5]) }
	],
	[#Rule 7
		 'spec', 4,
sub
#line 19 "share/bi.yp"
{ $_[0]->spec($_[1], [], $_[3]) }
	],
	[#Rule 8
		 'spec', 4,
sub
#line 20 "share/bi.yp"
{ $_[0]->spec($_[1], [], [], $_[3]) }
	],
	[#Rule 9
		 'spec', 3,
sub
#line 21 "share/bi.yp"
{ $_[0]->spec($_[1]) }
	],
	[#Rule 10
		 'spec', 1,
sub
#line 22 "share/bi.yp"
{ $_[0]->spec($_[1]) }
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
		 'model_definitions', 2,
sub
#line 52 "share/bi.yp"
{ $_[0]->append($_[1], $_[2]) }
	],
	[#Rule 23
		 'model_definitions', 1,
sub
#line 53 "share/bi.yp"
{ $_[0]->append($_[1]) }
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
		 'model_definition', 1, undef
	],
	[#Rule 30
		 'model_definition', 1, undef
	],
	[#Rule 31
		 'model_definition', 1, undef
	],
	[#Rule 32
		 'model_definition', 1, undef
	],
	[#Rule 33
		 'model_definition', 1, undef
	],
	[#Rule 34
		 'model_definition', 1, undef
	],
	[#Rule 35
		 'dim', 2,
sub
#line 71 "share/bi.yp"
{ $_[2] }
	],
	[#Rule 36
		 'dim_declarations', 3,
sub
#line 75 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 37
		 'dim_declarations', 1,
sub
#line 76 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 38
		 'dim_declaration', 2,
sub
#line 80 "share/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 39
		 'dim_declaration', 1,
sub
#line 81 "share/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 40
		 'const', 2,
sub
#line 85 "share/bi.yp"
{ $_[2] }
	],
	[#Rule 41
		 'const_declarations', 3,
sub
#line 89 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 42
		 'const_declarations', 1,
sub
#line 90 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 43
		 'const_declaration', 4,
sub
#line 94 "share/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 44
		 'const_declaration', 3,
sub
#line 95 "share/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 45
		 'inline', 2,
sub
#line 99 "share/bi.yp"
{ $_[2] }
	],
	[#Rule 46
		 'inline_declarations', 3,
sub
#line 103 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 47
		 'inline_declarations', 1,
sub
#line 104 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 48
		 'inline_declaration', 4,
sub
#line 108 "share/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 49
		 'inline_declaration', 3,
sub
#line 109 "share/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 50
		 'state', 3,
sub
#line 113 "share/bi.yp"
{ $_[0]->state($_[2]) }
	],
	[#Rule 51
		 'state', 2,
sub
#line 114 "share/bi.yp"
{ $_[0]->state($_[2]) }
	],
	[#Rule 52
		 'state_aux', 3,
sub
#line 118 "share/bi.yp"
{ $_[0]->state_aux($_[2]) }
	],
	[#Rule 53
		 'state_aux', 2,
sub
#line 119 "share/bi.yp"
{ $_[0]->state_aux($_[2]) }
	],
	[#Rule 54
		 'noise', 3,
sub
#line 123 "share/bi.yp"
{ $_[0]->noise($_[2]) }
	],
	[#Rule 55
		 'noise', 2,
sub
#line 124 "share/bi.yp"
{ $_[0]->noise($_[2]) }
	],
	[#Rule 56
		 'input', 3,
sub
#line 128 "share/bi.yp"
{ $_[0]->input($_[2]) }
	],
	[#Rule 57
		 'input', 2,
sub
#line 129 "share/bi.yp"
{ $_[0]->input($_[2]) }
	],
	[#Rule 58
		 'obs', 3,
sub
#line 133 "share/bi.yp"
{ $_[0]->obs($_[2]) }
	],
	[#Rule 59
		 'obs', 2,
sub
#line 134 "share/bi.yp"
{ $_[0]->obs($_[2]) }
	],
	[#Rule 60
		 'param', 3,
sub
#line 138 "share/bi.yp"
{ $_[0]->param($_[2]) }
	],
	[#Rule 61
		 'param', 2,
sub
#line 139 "share/bi.yp"
{ $_[0]->param($_[2]) }
	],
	[#Rule 62
		 'param_aux', 3,
sub
#line 143 "share/bi.yp"
{ $_[0]->param_aux($_[2]) }
	],
	[#Rule 63
		 'param_aux', 2,
sub
#line 144 "share/bi.yp"
{ $_[0]->param_aux($_[2]) }
	],
	[#Rule 64
		 'var_declarations', 3,
sub
#line 148 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 65
		 'var_declarations', 1,
sub
#line 149 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 66
		 'var_declaration', 1, undef
	],
	[#Rule 67
		 'var_declaration', 1, undef
	],
	[#Rule 68
		 'dim_args', 3,
sub
#line 158 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 69
		 'dim_args', 1,
sub
#line 159 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 70
		 'dim_arg', 1,
sub
#line 163 "share/bi.yp"
{ $_[0]->dim_arg($_[1]) }
	],
	[#Rule 71
		 'dim_aliases', 3,
sub
#line 167 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 72
		 'dim_aliases', 1,
sub
#line 168 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 73
		 'dim_alias', 1,
sub
#line 172 "share/bi.yp"
{ $_[0]->dim_alias($_[1]) }
	],
	[#Rule 74
		 'dim_ranges', 3,
sub
#line 176 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 75
		 'dim_ranges', 1,
sub
#line 177 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 76
		 'dim_range', 3,
sub
#line 181 "share/bi.yp"
{ $_[0]->dim_range($_[1], $_[3]) }
	],
	[#Rule 77
		 'dim_range', 1,
sub
#line 182 "share/bi.yp"
{ $_[0]->dim_range($_[1], $_[1]) }
	],
	[#Rule 78
		 'offset_args', 3,
sub
#line 186 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 79
		 'offset_args', 1,
sub
#line 187 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 80
		 'offset_arg', 1,
sub
#line 191 "share/bi.yp"
{ $_[0]->offset($_[1]) }
	],
	[#Rule 81
		 'offset_arg', 3,
sub
#line 192 "share/bi.yp"
{ $_[0]->offset($_[1], -1, $_[3]) }
	],
	[#Rule 82
		 'offset_arg', 3,
sub
#line 193 "share/bi.yp"
{ $_[0]->offset($_[1], 1, $_[3]) }
	],
	[#Rule 83
		 'offset_arg', 2,
sub
#line 194 "share/bi.yp"
{ $_[0]->offset($_[1], -1, $_[2]) }
	],
	[#Rule 84
		 'offset_arg', 1,
sub
#line 195 "share/bi.yp"
{ $_[0]->offset($_[1], 1, $_[2]) }
	],
	[#Rule 85
		 'range_args', 3,
sub
#line 199 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 86
		 'range_args', 1,
sub
#line 200 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 87
		 'range_arg', 3,
sub
#line 204 "share/bi.yp"
{ $_[0]->dim_range($_[1], $_[3]) }
	],
	[#Rule 88
		 'top_block', 2,
sub
#line 208 "share/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 89
		 'do', 2,
sub
#line 212 "share/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 90
		 'do', 3,
sub
#line 213 "share/bi.yp"
{ $_[0]->append($_[1], $_[0]->commit_block($_[3])) }
	],
	[#Rule 91
		 'block', 4,
sub
#line 217 "share/bi.yp"
{ $_[0]->block($_[1], $_[3]) }
	],
	[#Rule 92
		 'block', 3,
sub
#line 218 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 93
		 'block', 3,
sub
#line 219 "share/bi.yp"
{ $_[0]->block(undef, $_[2]) }
	],
	[#Rule 94
		 'block', 2,
sub
#line 220 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 95
		 'block_definitions', 2,
sub
#line 224 "share/bi.yp"
{ $_[0]->append($_[1], $_[2]) }
	],
	[#Rule 96
		 'block_definitions', 1,
sub
#line 225 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 97
		 'block_definition', 1, undef
	],
	[#Rule 98
		 'block_definition', 1, undef
	],
	[#Rule 99
		 'block_definition', 1, undef
	],
	[#Rule 100
		 'block_definition', 1, undef
	],
	[#Rule 101
		 'block_definition', 1, undef
	],
	[#Rule 102
		 'block_definition', 1, undef
	],
	[#Rule 103
		 'distributed_as', 7,
sub
#line 238 "share/bi.yp"
{ $_[0]->action($_[1], $_[3], undef, $_[5], $_[6]) }
	],
	[#Rule 104
		 'distributed_as', 6,
sub
#line 239 "share/bi.yp"
{ $_[0]->action($_[1], $_[3], undef, $_[5], $_[6]) }
	],
	[#Rule 105
		 'distributed_as', 7,
sub
#line 240 "share/bi.yp"
{ $_[0]->action($_[1], undef, $_[3], $_[5], $_[6]) }
	],
	[#Rule 106
		 'distributed_as', 6,
sub
#line 241 "share/bi.yp"
{ $_[0]->action($_[1], undef, $_[3], $_[5], $_[6]) }
	],
	[#Rule 107
		 'distributed_as', 4,
sub
#line 242 "share/bi.yp"
{ $_[0]->action($_[1], undef, undef, $_[2], $_[3]) }
	],
	[#Rule 108
		 'distributed_as', 3,
sub
#line 243 "share/bi.yp"
{ $_[0]->action($_[1], undef, undef, $_[2], $_[3]) }
	],
	[#Rule 109
		 'set_to', 7,
sub
#line 247 "share/bi.yp"
{ $_[0]->action($_[1], $_[3], undef, $_[5], $_[6]) }
	],
	[#Rule 110
		 'set_to', 6,
sub
#line 248 "share/bi.yp"
{ $_[0]->action($_[1], $_[3], undef, $_[5], $_[6]) }
	],
	[#Rule 111
		 'set_to', 7,
sub
#line 249 "share/bi.yp"
{ $_[0]->action($_[1], undef, $_[3], $_[5], $_[6]) }
	],
	[#Rule 112
		 'set_to', 6,
sub
#line 250 "share/bi.yp"
{ $_[0]->action($_[1], undef, $_[3], $_[5], $_[6]) }
	],
	[#Rule 113
		 'set_to', 4,
sub
#line 251 "share/bi.yp"
{ $_[0]->action($_[1], undef, undef, $_[2], $_[3]) }
	],
	[#Rule 114
		 'set_to', 3,
sub
#line 252 "share/bi.yp"
{ $_[0]->action($_[1], undef, undef, $_[2], $_[3]) }
	],
	[#Rule 115
		 'postfix_expression', 1,
sub
#line 261 "share/bi.yp"
{ $_[0]->literal($_[1]) }
	],
	[#Rule 116
		 'postfix_expression', 1,
sub
#line 262 "share/bi.yp"
{ $_[0]->literal($_[1]) }
	],
	[#Rule 117
		 'postfix_expression', 1,
sub
#line 263 "share/bi.yp"
{ $_[0]->string_literal($_[1]) }
	],
	[#Rule 118
		 'postfix_expression', 1,
sub
#line 264 "share/bi.yp"
{ $_[0]->identifier($_[1]) }
	],
	[#Rule 119
		 'postfix_expression', 4,
sub
#line 265 "share/bi.yp"
{ $_[0]->identifier($_[1], $_[3]) }
	],
	[#Rule 120
		 'postfix_expression', 4,
sub
#line 266 "share/bi.yp"
{ $_[0]->identifier($_[1], undef, $_[3]) }
	],
	[#Rule 121
		 'postfix_expression', 4,
sub
#line 267 "share/bi.yp"
{ $_[0]->function($_[1], $_[3]) }
	],
	[#Rule 122
		 'postfix_expression', 3,
sub
#line 268 "share/bi.yp"
{ $_[0]->function($_[1]) }
	],
	[#Rule 123
		 'postfix_expression', 3,
sub
#line 269 "share/bi.yp"
{ $_[0]->parens($_[2]) }
	],
	[#Rule 124
		 'unary_expression', 1,
sub
#line 273 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 125
		 'unary_expression', 2,
sub
#line 274 "share/bi.yp"
{ $_[0]->unary_operator($_[1], $_[2]) }
	],
	[#Rule 126
		 'unary_operator', 1,
sub
#line 280 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 127
		 'unary_operator', 1,
sub
#line 281 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 128
		 'unary_operator', 1,
sub
#line 283 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 129
		 'cast_expression', 1,
sub
#line 287 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 130
		 'multiplicative_expression', 1,
sub
#line 292 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 131
		 'multiplicative_expression', 3,
sub
#line 293 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 132
		 'multiplicative_expression', 3,
sub
#line 294 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 133
		 'multiplicative_expression', 3,
sub
#line 295 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 134
		 'multiplicative_expression', 3,
sub
#line 296 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 135
		 'additive_expression', 1,
sub
#line 301 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 136
		 'additive_expression', 3,
sub
#line 302 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 137
		 'additive_expression', 3,
sub
#line 303 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 138
		 'additive_expression', 3,
sub
#line 304 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 139
		 'additive_expression', 3,
sub
#line 305 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 140
		 'shift_expression', 1,
sub
#line 309 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 141
		 'relational_expression', 1,
sub
#line 315 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 142
		 'relational_expression', 3,
sub
#line 316 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 143
		 'relational_expression', 3,
sub
#line 317 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 144
		 'relational_expression', 3,
sub
#line 318 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 145
		 'relational_expression', 3,
sub
#line 319 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 146
		 'equality_expression', 1,
sub
#line 323 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 147
		 'equality_expression', 3,
sub
#line 324 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 148
		 'equality_expression', 3,
sub
#line 325 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 149
		 'and_expression', 1,
sub
#line 329 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 150
		 'exclusive_or_expression', 1,
sub
#line 334 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 151
		 'inclusive_or_expression', 1,
sub
#line 339 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 152
		 'logical_and_expression', 1,
sub
#line 344 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 153
		 'logical_and_expression', 3,
sub
#line 345 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 154
		 'logical_or_expression', 1,
sub
#line 349 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 155
		 'logical_or_expression', 3,
sub
#line 350 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 156
		 'conditional_expression', 1,
sub
#line 354 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 157
		 'conditional_expression', 5,
sub
#line 355 "share/bi.yp"
{ $_[0]->ternary_operator($_[1], $_[2], $_[3], $_[4], $_[5]) }
	],
	[#Rule 158
		 'expression', 1,
sub
#line 359 "share/bi.yp"
{ $_[0]->expression($_[1]) }
	]
],
                                  @_);
    bless($self,$class);
}

#line 362 "share/bi.yp"


1;

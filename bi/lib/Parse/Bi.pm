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
		DEFAULT => -116
	},
	{#State 12
		DEFAULT => -115
	},
	{#State 13
		DEFAULT => -147
	},
	{#State 14
		DEFAULT => -104
	},
	{#State 15
		DEFAULT => -103
	},
	{#State 16
		DEFAULT => -139
	},
	{#State 17
		DEFAULT => -141
	},
	{#State 18
		DEFAULT => -117
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
		DEFAULT => -118
	},
	{#State 22
		DEFAULT => -9
	},
	{#State 23
		ACTIONS => {
			'EQ_OP' => 67,
			'NE_OP' => 68
		},
		DEFAULT => -138
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
		DEFAULT => -106
	},
	{#State 26
		DEFAULT => -130
	},
	{#State 27
		ACTIONS => {
			"?" => 74,
			'OR_OP' => 75
		},
		DEFAULT => -145
	},
	{#State 28
		ACTIONS => {
			"-" => 76,
			'ELEM_ADD_OP' => 79,
			"+" => 77,
			'ELEM_SUB_OP' => 78
		},
		DEFAULT => -129
	},
	{#State 29
		DEFAULT => -17
	},
	{#State 30
		DEFAULT => -113
	},
	{#State 31
		DEFAULT => -20
	},
	{#State 32
		DEFAULT => -140
	},
	{#State 33
		ACTIONS => {
			"<" => 80,
			'LE_OP' => 81,
			'GE_OP' => 82,
			">" => 83
		},
		DEFAULT => -135
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
		DEFAULT => -143
	},
	{#State 36
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -124
	},
	{#State 37
		ACTIONS => {
			"," => 90,
			")" => 91
		}
	},
	{#State 38
		DEFAULT => -105
	},
	{#State 39
		DEFAULT => -119
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
		DEFAULT => -106
	},
	{#State 66
		DEFAULT => -114
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
			'expression' => 121,
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
			'index_args' => 122,
			'index_arg' => 123,
			'cast_expression' => 39
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
			'expression' => 124,
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
			'IDENTIFIER' => 25,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38,
			")" => 125
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'positional_args' => 126,
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
			'named_args' => 127,
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
			'conditional_expression' => 128,
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
			'logical_and_expression' => 129,
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
			'multiplicative_expression' => 130,
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
			'multiplicative_expression' => 131,
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
			'multiplicative_expression' => 132,
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
			'multiplicative_expression' => 133,
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
			'shift_expression' => 134,
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
			'shift_expression' => 135,
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
			'shift_expression' => 136,
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
			'shift_expression' => 137,
			'unary_expression' => 21,
			'postfix_expression' => 30,
			'additive_expression' => 28
		}
	},
	{#State 84
		ACTIONS => {
			")" => 138
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
			'inclusive_or_expression' => 139,
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
			'cast_expression' => 140,
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
			'cast_expression' => 141,
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
			'cast_expression' => 142,
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
			'cast_expression' => 143,
			'unary_expression' => 21,
			'postfix_expression' => 30
		}
	},
	{#State 90
		ACTIONS => {
			'IDENTIFIER' => 144
		},
		GOTOS => {
			'named_arg' => 145
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
			"[" => 146
		},
		DEFAULT => -10
	},
	{#State 95
		DEFAULT => -66
	},
	{#State 96
		ACTIONS => {
			";" => 147,
			"," => 148
		},
		DEFAULT => -57
	},
	{#State 97
		ACTIONS => {
			";" => 149,
			"," => 148
		},
		DEFAULT => -63
	},
	{#State 98
		ACTIONS => {
			";" => 150,
			"," => 148
		},
		DEFAULT => -59
	},
	{#State 99
		ACTIONS => {
			";" => 151,
			"," => 148
		},
		DEFAULT => -53
	},
	{#State 100
		ACTIONS => {
			";" => 152,
			"," => 148
		},
		DEFAULT => -55
	},
	{#State 101
		ACTIONS => {
			"{" => 153
		}
	},
	{#State 102
		ACTIONS => {
			"}" => 154,
			'IDENTIFIER' => 161,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 165
		},
		GOTOS => {
			'inline' => 160,
			'spec' => 101,
			'distributed_as' => 155,
			'do' => 156,
			'block_definition' => 162,
			'set_to' => 157,
			'target' => 158,
			'const' => 163,
			'block' => 164,
			'block_definitions' => 159
		}
	},
	{#State 103
		DEFAULT => -82
	},
	{#State 104
		DEFAULT => -47
	},
	{#State 105
		ACTIONS => {
			"=" => 166
		}
	},
	{#State 106
		ACTIONS => {
			"," => 167
		},
		DEFAULT => -45
	},
	{#State 107
		ACTIONS => {
			"," => 168
		},
		DEFAULT => -40
	},
	{#State 108
		DEFAULT => -42
	},
	{#State 109
		ACTIONS => {
			"=" => 169
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
			"," => 170
		},
		DEFAULT => -35
	},
	{#State 113
		ACTIONS => {
			";" => 171
		},
		DEFAULT => -39
	},
	{#State 114
		DEFAULT => -37
	},
	{#State 115
		ACTIONS => {
			";" => 172,
			"," => 148
		},
		DEFAULT => -51
	},
	{#State 116
		ACTIONS => {
			";" => 173,
			"," => 148
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
		DEFAULT => -136
	},
	{#State 118
		ACTIONS => {
			"<" => 80,
			'LE_OP' => 81,
			'GE_OP' => 82,
			">" => 83
		},
		DEFAULT => -137
	},
	{#State 119
		DEFAULT => -16
	},
	{#State 120
		ACTIONS => {
			"," => 90,
			")" => 174
		}
	},
	{#State 121
		ACTIONS => {
			":" => 175
		},
		DEFAULT => -80
	},
	{#State 122
		ACTIONS => {
			"," => 176,
			"]" => 177
		}
	},
	{#State 123
		DEFAULT => -79
	},
	{#State 124
		DEFAULT => -21
	},
	{#State 125
		DEFAULT => -111
	},
	{#State 126
		ACTIONS => {
			"," => 178,
			")" => 179
		}
	},
	{#State 127
		ACTIONS => {
			"," => 90,
			")" => 180
		}
	},
	{#State 128
		ACTIONS => {
			":" => 181
		}
	},
	{#State 129
		ACTIONS => {
			'AND_OP' => 85
		},
		DEFAULT => -144
	},
	{#State 130
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -127
	},
	{#State 131
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -125
	},
	{#State 132
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -128
	},
	{#State 133
		ACTIONS => {
			"*" => 86,
			'ELEM_MUL_OP' => 88,
			"/" => 87,
			'ELEM_DIV_OP' => 89
		},
		DEFAULT => -126
	},
	{#State 134
		DEFAULT => -131
	},
	{#State 135
		DEFAULT => -133
	},
	{#State 136
		DEFAULT => -134
	},
	{#State 137
		DEFAULT => -132
	},
	{#State 138
		DEFAULT => -112
	},
	{#State 139
		DEFAULT => -142
	},
	{#State 140
		DEFAULT => -120
	},
	{#State 141
		DEFAULT => -122
	},
	{#State 142
		DEFAULT => -121
	},
	{#State 143
		DEFAULT => -123
	},
	{#State 144
		ACTIONS => {
			"=" => 72
		}
	},
	{#State 145
		DEFAULT => -19
	},
	{#State 146
		ACTIONS => {
			'IDENTIFIER' => 184
		},
		GOTOS => {
			'dim_args' => 182,
			'dim_arg' => 183
		}
	},
	{#State 147
		DEFAULT => -56
	},
	{#State 148
		ACTIONS => {
			'IDENTIFIER' => 94
		},
		GOTOS => {
			'var_declaration' => 185,
			'spec' => 92,
			'array_spec' => 95
		}
	},
	{#State 149
		DEFAULT => -62
	},
	{#State 150
		DEFAULT => -58
	},
	{#State 151
		DEFAULT => -52
	},
	{#State 152
		DEFAULT => -54
	},
	{#State 153
		ACTIONS => {
			"}" => 186,
			'IDENTIFIER' => 161,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 165
		},
		GOTOS => {
			'inline' => 160,
			'spec' => 101,
			'distributed_as' => 155,
			'do' => 156,
			'block_definition' => 162,
			'set_to' => 157,
			'target' => 158,
			'const' => 163,
			'block' => 164,
			'block_definitions' => 187
		}
	},
	{#State 154
		DEFAULT => -88
	},
	{#State 155
		DEFAULT => -92
	},
	{#State 156
		ACTIONS => {
			'THEN' => 188
		},
		DEFAULT => -91
	},
	{#State 157
		DEFAULT => -93
	},
	{#State 158
		ACTIONS => {
			"~" => 189,
			'SET_TO' => 190
		}
	},
	{#State 159
		ACTIONS => {
			"}" => 191,
			'IDENTIFIER' => 161,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 165
		},
		GOTOS => {
			'inline' => 160,
			'spec' => 101,
			'distributed_as' => 155,
			'do' => 156,
			'block_definition' => 192,
			'set_to' => 157,
			'target' => 158,
			'const' => 163,
			'block' => 164
		}
	},
	{#State 160
		DEFAULT => -95
	},
	{#State 161
		ACTIONS => {
			"(" => 9,
			"{" => -10,
			"[" => 193
		},
		DEFAULT => -102
	},
	{#State 162
		DEFAULT => -90
	},
	{#State 163
		DEFAULT => -94
	},
	{#State 164
		DEFAULT => -96
	},
	{#State 165
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 102
		},
		GOTOS => {
			'spec' => 101,
			'block' => 194
		}
	},
	{#State 166
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
			'expression' => 195,
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
	{#State 167
		ACTIONS => {
			'IDENTIFIER' => 105
		},
		GOTOS => {
			'inline_declaration' => 196
		}
	},
	{#State 168
		ACTIONS => {
			'IDENTIFIER' => 109
		},
		GOTOS => {
			'const_declaration' => 197
		}
	},
	{#State 169
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
			'expression' => 198,
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
	{#State 170
		ACTIONS => {
			'IDENTIFIER' => 7
		},
		GOTOS => {
			'spec' => 113,
			'dim_declaration' => 199
		}
	},
	{#State 171
		DEFAULT => -38
	},
	{#State 172
		DEFAULT => -50
	},
	{#State 173
		DEFAULT => -60
	},
	{#State 174
		DEFAULT => -6
	},
	{#State 175
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
			'expression' => 200,
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
	{#State 176
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
			'expression' => 121,
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
			'index_arg' => 201,
			'cast_expression' => 39
		}
	},
	{#State 177
		DEFAULT => -107
	},
	{#State 178
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
			'named_args' => 202,
			'cast_expression' => 39
		}
	},
	{#State 179
		DEFAULT => -109
	},
	{#State 180
		DEFAULT => -110
	},
	{#State 181
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
			'conditional_expression' => 203,
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
	{#State 182
		ACTIONS => {
			"," => 204,
			"]" => 205
		}
	},
	{#State 183
		DEFAULT => -69
	},
	{#State 184
		DEFAULT => -70
	},
	{#State 185
		DEFAULT => -64
	},
	{#State 186
		DEFAULT => -86
	},
	{#State 187
		ACTIONS => {
			"}" => 206,
			'IDENTIFIER' => 161,
			"{" => 102,
			'CONST' => 55,
			'INLINE' => 54,
			'DO' => 165
		},
		GOTOS => {
			'inline' => 160,
			'spec' => 101,
			'distributed_as' => 155,
			'do' => 156,
			'block_definition' => 192,
			'set_to' => 157,
			'target' => 158,
			'const' => 163,
			'block' => 164
		}
	},
	{#State 188
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 102
		},
		GOTOS => {
			'spec' => 101,
			'block' => 207
		}
	},
	{#State 189
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
			'expression' => 208,
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
	{#State 190
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
			'expression' => 209,
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
	{#State 191
		DEFAULT => -87
	},
	{#State 192
		DEFAULT => -89
	},
	{#State 193
		ACTIONS => {
			'IDENTIFIER' => 212,
			'INTEGER_LITERAL' => 210
		},
		GOTOS => {
			'dim_aliases' => 211,
			'dim_alias' => 213
		}
	},
	{#State 194
		DEFAULT => -83
	},
	{#State 195
		ACTIONS => {
			";" => 214
		},
		DEFAULT => -49
	},
	{#State 196
		DEFAULT => -46
	},
	{#State 197
		DEFAULT => -41
	},
	{#State 198
		ACTIONS => {
			";" => 215
		},
		DEFAULT => -44
	},
	{#State 199
		DEFAULT => -36
	},
	{#State 200
		DEFAULT => -81
	},
	{#State 201
		DEFAULT => -78
	},
	{#State 202
		ACTIONS => {
			"," => 90,
			")" => 216
		}
	},
	{#State 203
		DEFAULT => -146
	},
	{#State 204
		ACTIONS => {
			'IDENTIFIER' => 184
		},
		GOTOS => {
			'dim_arg' => 217
		}
	},
	{#State 205
		ACTIONS => {
			"(" => 218
		},
		DEFAULT => -15
	},
	{#State 206
		DEFAULT => -85
	},
	{#State 207
		DEFAULT => -84
	},
	{#State 208
		ACTIONS => {
			";" => 219
		},
		DEFAULT => -98
	},
	{#State 209
		ACTIONS => {
			";" => 220
		},
		DEFAULT => -100
	},
	{#State 210
		ACTIONS => {
			":" => 221
		},
		DEFAULT => -76
	},
	{#State 211
		ACTIONS => {
			"," => 222,
			"]" => 223
		}
	},
	{#State 212
		ACTIONS => {
			"=" => 224
		},
		DEFAULT => -73
	},
	{#State 213
		DEFAULT => -72
	},
	{#State 214
		DEFAULT => -48
	},
	{#State 215
		DEFAULT => -43
	},
	{#State 216
		DEFAULT => -108
	},
	{#State 217
		DEFAULT => -68
	},
	{#State 218
		ACTIONS => {
			"-" => 11,
			'IDENTIFIER' => 25,
			"+" => 12,
			'INTEGER_LITERAL' => 14,
			'LITERAL' => 15,
			"!" => 18,
			"(" => 34,
			'STRING_LITERAL' => 38,
			")" => 225
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 16,
			'inclusive_or_expression' => 17,
			'expression' => 19,
			'unary_operator' => 20,
			'unary_expression' => 21,
			'equality_expression' => 23,
			'positional_args' => 226,
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
			'named_args' => 227,
			'cast_expression' => 39
		}
	},
	{#State 219
		DEFAULT => -97
	},
	{#State 220
		DEFAULT => -99
	},
	{#State 221
		ACTIONS => {
			'INTEGER_LITERAL' => 228
		}
	},
	{#State 222
		ACTIONS => {
			'IDENTIFIER' => 212,
			'INTEGER_LITERAL' => 210
		},
		GOTOS => {
			'dim_alias' => 229
		}
	},
	{#State 223
		DEFAULT => -101
	},
	{#State 224
		ACTIONS => {
			'INTEGER_LITERAL' => 230
		}
	},
	{#State 225
		DEFAULT => -14
	},
	{#State 226
		ACTIONS => {
			"," => 231,
			")" => 232
		}
	},
	{#State 227
		ACTIONS => {
			"," => 90,
			")" => 233
		}
	},
	{#State 228
		DEFAULT => -77
	},
	{#State 229
		DEFAULT => -71
	},
	{#State 230
		ACTIONS => {
			":" => 234
		},
		DEFAULT => -74
	},
	{#State 231
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
			'named_args' => 235,
			'cast_expression' => 39
		}
	},
	{#State 232
		DEFAULT => -12
	},
	{#State 233
		DEFAULT => -13
	},
	{#State 234
		ACTIONS => {
			'INTEGER_LITERAL' => 236
		}
	},
	{#State 235
		ACTIONS => {
			"," => 90,
			")" => 237
		}
	},
	{#State 236
		DEFAULT => -75
	},
	{#State 237
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
		 'dim_alias', 3,
sub
#line 173 "share/bi.yp"
{ $_[0]->dim_alias($_[1], $_[3]) }
	],
	[#Rule 75
		 'dim_alias', 5,
sub
#line 174 "share/bi.yp"
{ $_[0]->dim_alias($_[1], $_[3], $_[5]) }
	],
	[#Rule 76
		 'dim_alias', 1,
sub
#line 175 "share/bi.yp"
{ $_[0]->dim_alias(undef, $_[3]) }
	],
	[#Rule 77
		 'dim_alias', 3,
sub
#line 176 "share/bi.yp"
{ $_[0]->dim_alias(undef, $_[3], $_[5]) }
	],
	[#Rule 78
		 'index_args', 3,
sub
#line 180 "share/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 79
		 'index_args', 1,
sub
#line 181 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 80
		 'index_arg', 1,
sub
#line 185 "share/bi.yp"
{ $_[0]->index($_[1]) }
	],
	[#Rule 81
		 'index_arg', 3,
sub
#line 186 "share/bi.yp"
{ $_[0]->range($_[1], $_[3]) }
	],
	[#Rule 82
		 'top_block', 2,
sub
#line 190 "share/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 83
		 'do', 2,
sub
#line 194 "share/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 84
		 'do', 3,
sub
#line 195 "share/bi.yp"
{ $_[0]->append($_[1], $_[0]->commit_block($_[3])) }
	],
	[#Rule 85
		 'block', 4,
sub
#line 199 "share/bi.yp"
{ $_[0]->block($_[1], $_[3]) }
	],
	[#Rule 86
		 'block', 3,
sub
#line 200 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 87
		 'block', 3,
sub
#line 201 "share/bi.yp"
{ $_[0]->block(undef, $_[2]) }
	],
	[#Rule 88
		 'block', 2,
sub
#line 202 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 89
		 'block_definitions', 2,
sub
#line 206 "share/bi.yp"
{ $_[0]->append($_[1], $_[2]) }
	],
	[#Rule 90
		 'block_definitions', 1,
sub
#line 207 "share/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 91
		 'block_definition', 1, undef
	],
	[#Rule 92
		 'block_definition', 1, undef
	],
	[#Rule 93
		 'block_definition', 1, undef
	],
	[#Rule 94
		 'block_definition', 1, undef
	],
	[#Rule 95
		 'block_definition', 1, undef
	],
	[#Rule 96
		 'block_definition', 1, undef
	],
	[#Rule 97
		 'distributed_as', 4,
sub
#line 220 "share/bi.yp"
{ $_[0]->action($_[1], $_[2], $_[3]) }
	],
	[#Rule 98
		 'distributed_as', 3,
sub
#line 221 "share/bi.yp"
{ $_[0]->action($_[1], $_[2], $_[3]) }
	],
	[#Rule 99
		 'set_to', 4,
sub
#line 225 "share/bi.yp"
{ $_[0]->action($_[1], $_[2], $_[3]) }
	],
	[#Rule 100
		 'set_to', 3,
sub
#line 226 "share/bi.yp"
{ $_[0]->action($_[1], $_[2], $_[3]) }
	],
	[#Rule 101
		 'target', 4,
sub
#line 230 "share/bi.yp"
{ $_[0]->target($_[1], $_[3]) }
	],
	[#Rule 102
		 'target', 1,
sub
#line 231 "share/bi.yp"
{ $_[0]->target($_[1]) }
	],
	[#Rule 103
		 'postfix_expression', 1,
sub
#line 240 "share/bi.yp"
{ $_[0]->literal($_[1]) }
	],
	[#Rule 104
		 'postfix_expression', 1,
sub
#line 241 "share/bi.yp"
{ $_[0]->integer_literal($_[1]) }
	],
	[#Rule 105
		 'postfix_expression', 1,
sub
#line 242 "share/bi.yp"
{ $_[0]->string_literal($_[1]) }
	],
	[#Rule 106
		 'postfix_expression', 1,
sub
#line 243 "share/bi.yp"
{ $_[0]->identifier($_[1]) }
	],
	[#Rule 107
		 'postfix_expression', 4,
sub
#line 244 "share/bi.yp"
{ $_[0]->identifier($_[1], $_[3]) }
	],
	[#Rule 108
		 'postfix_expression', 6,
sub
#line 245 "share/bi.yp"
{ $_[0]->function($_[1], $_[3], $_[5]) }
	],
	[#Rule 109
		 'postfix_expression', 4,
sub
#line 246 "share/bi.yp"
{ $_[0]->function($_[1], $_[3]) }
	],
	[#Rule 110
		 'postfix_expression', 4,
sub
#line 247 "share/bi.yp"
{ $_[0]->function($_[1], undef, $_[3]) }
	],
	[#Rule 111
		 'postfix_expression', 3,
sub
#line 248 "share/bi.yp"
{ $_[0]->function($_[1]) }
	],
	[#Rule 112
		 'postfix_expression', 3,
sub
#line 249 "share/bi.yp"
{ $_[0]->parens($_[2]) }
	],
	[#Rule 113
		 'unary_expression', 1,
sub
#line 253 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 114
		 'unary_expression', 2,
sub
#line 254 "share/bi.yp"
{ $_[0]->unary_operator($_[1], $_[2]) }
	],
	[#Rule 115
		 'unary_operator', 1,
sub
#line 260 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 116
		 'unary_operator', 1,
sub
#line 261 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 117
		 'unary_operator', 1,
sub
#line 263 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 118
		 'cast_expression', 1,
sub
#line 267 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 119
		 'multiplicative_expression', 1,
sub
#line 272 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 120
		 'multiplicative_expression', 3,
sub
#line 273 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 121
		 'multiplicative_expression', 3,
sub
#line 274 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 122
		 'multiplicative_expression', 3,
sub
#line 275 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 123
		 'multiplicative_expression', 3,
sub
#line 276 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 124
		 'additive_expression', 1,
sub
#line 281 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 125
		 'additive_expression', 3,
sub
#line 282 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 126
		 'additive_expression', 3,
sub
#line 283 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 127
		 'additive_expression', 3,
sub
#line 284 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 128
		 'additive_expression', 3,
sub
#line 285 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 129
		 'shift_expression', 1,
sub
#line 289 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 130
		 'relational_expression', 1,
sub
#line 295 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 131
		 'relational_expression', 3,
sub
#line 296 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 132
		 'relational_expression', 3,
sub
#line 297 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 133
		 'relational_expression', 3,
sub
#line 298 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 134
		 'relational_expression', 3,
sub
#line 299 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 135
		 'equality_expression', 1,
sub
#line 303 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 136
		 'equality_expression', 3,
sub
#line 304 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 137
		 'equality_expression', 3,
sub
#line 305 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 138
		 'and_expression', 1,
sub
#line 309 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 139
		 'exclusive_or_expression', 1,
sub
#line 314 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 140
		 'inclusive_or_expression', 1,
sub
#line 319 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 141
		 'logical_and_expression', 1,
sub
#line 324 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 142
		 'logical_and_expression', 3,
sub
#line 325 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 143
		 'logical_or_expression', 1,
sub
#line 329 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 144
		 'logical_or_expression', 3,
sub
#line 330 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 145
		 'conditional_expression', 1,
sub
#line 334 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 146
		 'conditional_expression', 5,
sub
#line 335 "share/bi.yp"
{ $_[0]->ternary_operator($_[1], $_[2], $_[3], $_[4], $_[5]) }
	],
	[#Rule 147
		 'expression', 1,
sub
#line 339 "share/bi.yp"
{ $_[0]->expression($_[1]) }
	]
],
                                  @_);
    bless($self,$class);
}

#line 342 "share/bi.yp"


1;

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
			"(" => 33,
			'IDENTIFIER' => 24,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14,
			")" => 21
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
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
			'named_args' => 37,
			'cast_expression' => 38
		}
	},
	{#State 10
		ACTIONS => {
			'NOISE' => 50,
			"}" => 39,
			'INPUT' => 40,
			'SUB' => 52,
			'CONST' => 54,
			'INLINE' => 53,
			'PARAM_AUX' => 42,
			'DIM' => 57,
			'STATE' => 58,
			'OBS' => 45,
			'PARAM' => 61,
			'STATE_AUX' => 49
		},
		GOTOS => {
			'inline' => 51,
			'input' => 41,
			'state' => 43,
			'param_aux' => 44,
			'model_definitions' => 55,
			'const' => 56,
			'obs' => 60,
			'top_block' => 59,
			'dim' => 46,
			'model_definition' => 47,
			'param' => 48,
			'state_aux' => 63,
			'noise' => 62
		}
	},
	{#State 11
		DEFAULT => -114
	},
	{#State 12
		DEFAULT => -113
	},
	{#State 13
		DEFAULT => -145
	},
	{#State 14
		DEFAULT => -104
	},
	{#State 15
		DEFAULT => -137
	},
	{#State 16
		DEFAULT => -139
	},
	{#State 17
		DEFAULT => -115
	},
	{#State 18
		DEFAULT => -18
	},
	{#State 19
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 65,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 20
		DEFAULT => -116
	},
	{#State 21
		DEFAULT => -9
	},
	{#State 22
		ACTIONS => {
			'EQ_OP' => 66,
			'NE_OP' => 67
		},
		DEFAULT => -136
	},
	{#State 23
		ACTIONS => {
			"," => 68,
			")" => 69
		}
	},
	{#State 24
		ACTIONS => {
			"[" => 70,
			"=" => 71,
			"(" => 72
		},
		DEFAULT => -106
	},
	{#State 25
		DEFAULT => -128
	},
	{#State 26
		ACTIONS => {
			"?" => 73,
			'OR_OP' => 74
		},
		DEFAULT => -143
	},
	{#State 27
		ACTIONS => {
			"-" => 75,
			'ELEM_ADD_OP' => 78,
			"+" => 76,
			'ELEM_SUB_OP' => 77
		},
		DEFAULT => -127
	},
	{#State 28
		DEFAULT => -17
	},
	{#State 29
		DEFAULT => -111
	},
	{#State 30
		DEFAULT => -20
	},
	{#State 31
		DEFAULT => -138
	},
	{#State 32
		ACTIONS => {
			"<" => 79,
			'LE_OP' => 80,
			'GE_OP' => 81,
			">" => 82
		},
		DEFAULT => -133
	},
	{#State 33
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'equality_expression' => 22,
			'conditional_expression' => 83,
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'logical_or_expression' => 26,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 34
		ACTIONS => {
			'AND_OP' => 84
		},
		DEFAULT => -141
	},
	{#State 35
		ACTIONS => {
			"*" => 85,
			'ELEM_MUL_OP' => 87,
			"/" => 86,
			'ELEM_DIV_OP' => 88
		},
		DEFAULT => -122
	},
	{#State 36
		DEFAULT => -105
	},
	{#State 37
		ACTIONS => {
			"," => 89,
			")" => 90
		}
	},
	{#State 38
		DEFAULT => -117
	},
	{#State 39
		DEFAULT => -5
	},
	{#State 40
		ACTIONS => {
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 92,
			'spec' => 91,
			'array_spec' => 94,
			'var_declarations' => 95
		}
	},
	{#State 41
		DEFAULT => -28
	},
	{#State 42
		ACTIONS => {
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 92,
			'spec' => 91,
			'array_spec' => 94,
			'var_declarations' => 96
		}
	},
	{#State 43
		DEFAULT => -25
	},
	{#State 44
		DEFAULT => -31
	},
	{#State 45
		ACTIONS => {
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 92,
			'spec' => 91,
			'array_spec' => 94,
			'var_declarations' => 97
		}
	},
	{#State 46
		DEFAULT => -24
	},
	{#State 47
		DEFAULT => -23
	},
	{#State 48
		DEFAULT => -30
	},
	{#State 49
		ACTIONS => {
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 92,
			'spec' => 91,
			'array_spec' => 94,
			'var_declarations' => 98
		}
	},
	{#State 50
		ACTIONS => {
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 92,
			'spec' => 91,
			'array_spec' => 94,
			'var_declarations' => 99
		}
	},
	{#State 51
		DEFAULT => -33
	},
	{#State 52
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 101
		},
		GOTOS => {
			'spec' => 100,
			'block' => 102
		}
	},
	{#State 53
		ACTIONS => {
			'IDENTIFIER' => 104
		},
		GOTOS => {
			'inline_declaration' => 103,
			'inline_declarations' => 105
		}
	},
	{#State 54
		ACTIONS => {
			'IDENTIFIER' => 108
		},
		GOTOS => {
			'const_declarations' => 106,
			'const_declaration' => 107
		}
	},
	{#State 55
		ACTIONS => {
			'NOISE' => 50,
			"}" => 109,
			'INPUT' => 40,
			'SUB' => 52,
			'CONST' => 54,
			'INLINE' => 53,
			'PARAM_AUX' => 42,
			'DIM' => 57,
			'STATE' => 58,
			'OBS' => 45,
			'PARAM' => 61,
			'STATE_AUX' => 49
		},
		GOTOS => {
			'inline' => 51,
			'input' => 41,
			'state' => 43,
			'param_aux' => 44,
			'const' => 56,
			'obs' => 60,
			'top_block' => 59,
			'dim' => 46,
			'model_definition' => 110,
			'param' => 48,
			'state_aux' => 63,
			'noise' => 62
		}
	},
	{#State 56
		DEFAULT => -32
	},
	{#State 57
		ACTIONS => {
			'IDENTIFIER' => 7
		},
		GOTOS => {
			'spec' => 112,
			'dim_declarations' => 111,
			'dim_declaration' => 113
		}
	},
	{#State 58
		ACTIONS => {
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 92,
			'spec' => 91,
			'array_spec' => 94,
			'var_declarations' => 114
		}
	},
	{#State 59
		DEFAULT => -34
	},
	{#State 60
		DEFAULT => -29
	},
	{#State 61
		ACTIONS => {
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 92,
			'spec' => 91,
			'array_spec' => 94,
			'var_declarations' => 115
		}
	},
	{#State 62
		DEFAULT => -27
	},
	{#State 63
		DEFAULT => -26
	},
	{#State 64
		ACTIONS => {
			"[" => 70,
			"(" => 72
		},
		DEFAULT => -106
	},
	{#State 65
		DEFAULT => -112
	},
	{#State 66
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'relational_expression' => 116,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'cast_expression' => 38
		}
	},
	{#State 67
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'relational_expression' => 117,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'cast_expression' => 38
		}
	},
	{#State 68
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 24,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 118,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'named_args' => 119,
			'cast_expression' => 38
		}
	},
	{#State 69
		DEFAULT => -7
	},
	{#State 70
		ACTIONS => {
			"-" => 120,
			'IDENTIFIER' => 123,
			'LITERAL' => 121
		},
		GOTOS => {
			'offset_arg' => 124,
			'offset_args' => 122
		}
	},
	{#State 71
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 125,
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
	{#State 72
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14,
			")" => 126
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'positional_args' => 127,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 28,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'cast_expression' => 38
		}
	},
	{#State 73
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'equality_expression' => 22,
			'conditional_expression' => 128,
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'logical_or_expression' => 26,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 74
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'equality_expression' => 22,
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 129,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 75
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 130,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 76
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 131,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 77
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 132,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 78
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 133,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 79
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'shift_expression' => 134,
			'unary_expression' => 20,
			'postfix_expression' => 29,
			'additive_expression' => 27
		}
	},
	{#State 80
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'shift_expression' => 135,
			'unary_expression' => 20,
			'postfix_expression' => 29,
			'additive_expression' => 27
		}
	},
	{#State 81
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'shift_expression' => 136,
			'unary_expression' => 20,
			'postfix_expression' => 29,
			'additive_expression' => 27
		}
	},
	{#State 82
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'shift_expression' => 137,
			'unary_expression' => 20,
			'postfix_expression' => 29,
			'additive_expression' => 27
		}
	},
	{#State 83
		ACTIONS => {
			")" => 138
		}
	},
	{#State 84
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'equality_expression' => 22,
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'and_expression' => 15,
			'inclusive_or_expression' => 139,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
		}
	},
	{#State 85
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 140,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 86
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 141,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 87
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 142,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 88
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'unary_operator' => 19,
			'cast_expression' => 143,
			'unary_expression' => 20,
			'postfix_expression' => 29
		}
	},
	{#State 89
		ACTIONS => {
			'IDENTIFIER' => 144
		},
		GOTOS => {
			'named_arg' => 145
		}
	},
	{#State 90
		DEFAULT => -8
	},
	{#State 91
		DEFAULT => -67
	},
	{#State 92
		DEFAULT => -65
	},
	{#State 93
		ACTIONS => {
			"(" => 9,
			"[" => 146
		},
		DEFAULT => -10
	},
	{#State 94
		DEFAULT => -66
	},
	{#State 95
		ACTIONS => {
			";" => 147,
			"," => 148
		},
		DEFAULT => -57
	},
	{#State 96
		ACTIONS => {
			";" => 149,
			"," => 148
		},
		DEFAULT => -63
	},
	{#State 97
		ACTIONS => {
			";" => 150,
			"," => 148
		},
		DEFAULT => -59
	},
	{#State 98
		ACTIONS => {
			";" => 151,
			"," => 148
		},
		DEFAULT => -53
	},
	{#State 99
		ACTIONS => {
			";" => 152,
			"," => 148
		},
		DEFAULT => -55
	},
	{#State 100
		ACTIONS => {
			"{" => 153
		}
	},
	{#State 101
		ACTIONS => {
			"}" => 154,
			'IDENTIFIER' => 160,
			"{" => 101,
			'CONST' => 54,
			'INLINE' => 53,
			'DO' => 164
		},
		GOTOS => {
			'inline' => 159,
			'spec' => 100,
			'distributed_as' => 155,
			'do' => 156,
			'block_definition' => 161,
			'set_to' => 157,
			'const' => 162,
			'block' => 163,
			'block_definitions' => 158
		}
	},
	{#State 102
		DEFAULT => -81
	},
	{#State 103
		DEFAULT => -47
	},
	{#State 104
		ACTIONS => {
			"=" => 165
		}
	},
	{#State 105
		ACTIONS => {
			"," => 166
		},
		DEFAULT => -45
	},
	{#State 106
		ACTIONS => {
			"," => 167
		},
		DEFAULT => -40
	},
	{#State 107
		DEFAULT => -42
	},
	{#State 108
		ACTIONS => {
			"=" => 168
		}
	},
	{#State 109
		DEFAULT => -4
	},
	{#State 110
		DEFAULT => -22
	},
	{#State 111
		ACTIONS => {
			"," => 169
		},
		DEFAULT => -35
	},
	{#State 112
		ACTIONS => {
			";" => 170
		},
		DEFAULT => -39
	},
	{#State 113
		DEFAULT => -37
	},
	{#State 114
		ACTIONS => {
			";" => 171,
			"," => 148
		},
		DEFAULT => -51
	},
	{#State 115
		ACTIONS => {
			";" => 172,
			"," => 148
		},
		DEFAULT => -61
	},
	{#State 116
		ACTIONS => {
			"<" => 79,
			'LE_OP' => 80,
			'GE_OP' => 81,
			">" => 82
		},
		DEFAULT => -134
	},
	{#State 117
		ACTIONS => {
			"<" => 79,
			'LE_OP' => 80,
			'GE_OP' => 81,
			">" => 82
		},
		DEFAULT => -135
	},
	{#State 118
		DEFAULT => -16
	},
	{#State 119
		ACTIONS => {
			"," => 89,
			")" => 173
		}
	},
	{#State 120
		ACTIONS => {
			'LITERAL' => 174
		}
	},
	{#State 121
		DEFAULT => -80
	},
	{#State 122
		ACTIONS => {
			"," => 175,
			"]" => 176
		}
	},
	{#State 123
		ACTIONS => {
			"-" => 177,
			"+" => 178
		},
		DEFAULT => -76
	},
	{#State 124
		DEFAULT => -75
	},
	{#State 125
		DEFAULT => -21
	},
	{#State 126
		DEFAULT => -109
	},
	{#State 127
		ACTIONS => {
			"," => 179,
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
			'AND_OP' => 84
		},
		DEFAULT => -142
	},
	{#State 130
		ACTIONS => {
			"*" => 85,
			'ELEM_MUL_OP' => 87,
			"/" => 86,
			'ELEM_DIV_OP' => 88
		},
		DEFAULT => -125
	},
	{#State 131
		ACTIONS => {
			"*" => 85,
			'ELEM_MUL_OP' => 87,
			"/" => 86,
			'ELEM_DIV_OP' => 88
		},
		DEFAULT => -123
	},
	{#State 132
		ACTIONS => {
			"*" => 85,
			'ELEM_MUL_OP' => 87,
			"/" => 86,
			'ELEM_DIV_OP' => 88
		},
		DEFAULT => -126
	},
	{#State 133
		ACTIONS => {
			"*" => 85,
			'ELEM_MUL_OP' => 87,
			"/" => 86,
			'ELEM_DIV_OP' => 88
		},
		DEFAULT => -124
	},
	{#State 134
		DEFAULT => -129
	},
	{#State 135
		DEFAULT => -131
	},
	{#State 136
		DEFAULT => -132
	},
	{#State 137
		DEFAULT => -130
	},
	{#State 138
		DEFAULT => -110
	},
	{#State 139
		DEFAULT => -140
	},
	{#State 140
		DEFAULT => -118
	},
	{#State 141
		DEFAULT => -120
	},
	{#State 142
		DEFAULT => -119
	},
	{#State 143
		DEFAULT => -121
	},
	{#State 144
		ACTIONS => {
			"=" => 71
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
			'IDENTIFIER' => 93
		},
		GOTOS => {
			'var_declaration' => 185,
			'spec' => 91,
			'array_spec' => 94
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
			'IDENTIFIER' => 160,
			"{" => 101,
			'CONST' => 54,
			'INLINE' => 53,
			'DO' => 164
		},
		GOTOS => {
			'inline' => 159,
			'spec' => 100,
			'distributed_as' => 155,
			'do' => 156,
			'block_definition' => 161,
			'set_to' => 157,
			'const' => 162,
			'block' => 163,
			'block_definitions' => 187
		}
	},
	{#State 154
		DEFAULT => -87
	},
	{#State 155
		DEFAULT => -91
	},
	{#State 156
		ACTIONS => {
			'THEN' => 188
		},
		DEFAULT => -90
	},
	{#State 157
		DEFAULT => -92
	},
	{#State 158
		ACTIONS => {
			"}" => 189,
			'IDENTIFIER' => 160,
			"{" => 101,
			'CONST' => 54,
			'INLINE' => 53,
			'DO' => 164
		},
		GOTOS => {
			'inline' => 159,
			'spec' => 100,
			'const' => 162,
			'distributed_as' => 155,
			'do' => 156,
			'block' => 163,
			'block_definition' => 190,
			'set_to' => 157
		}
	},
	{#State 159
		DEFAULT => -94
	},
	{#State 160
		ACTIONS => {
			"(" => 9,
			"~" => 191,
			"[" => 192,
			'SET_TO' => 193
		},
		DEFAULT => -10
	},
	{#State 161
		DEFAULT => -89
	},
	{#State 162
		DEFAULT => -93
	},
	{#State 163
		DEFAULT => -95
	},
	{#State 164
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 101
		},
		GOTOS => {
			'spec' => 100,
			'block' => 194
		}
	},
	{#State 165
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 195,
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
	{#State 166
		ACTIONS => {
			'IDENTIFIER' => 104
		},
		GOTOS => {
			'inline_declaration' => 196
		}
	},
	{#State 167
		ACTIONS => {
			'IDENTIFIER' => 108
		},
		GOTOS => {
			'const_declaration' => 197
		}
	},
	{#State 168
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 198,
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
	{#State 169
		ACTIONS => {
			'IDENTIFIER' => 7
		},
		GOTOS => {
			'spec' => 112,
			'dim_declaration' => 199
		}
	},
	{#State 170
		DEFAULT => -38
	},
	{#State 171
		DEFAULT => -50
	},
	{#State 172
		DEFAULT => -60
	},
	{#State 173
		DEFAULT => -6
	},
	{#State 174
		DEFAULT => -79
	},
	{#State 175
		ACTIONS => {
			"-" => 120,
			'IDENTIFIER' => 123,
			'LITERAL' => 121
		},
		GOTOS => {
			'offset_arg' => 200
		}
	},
	{#State 176
		DEFAULT => -107
	},
	{#State 177
		ACTIONS => {
			'LITERAL' => 201
		}
	},
	{#State 178
		ACTIONS => {
			'LITERAL' => 202
		}
	},
	{#State 179
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 118,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'cast_expression' => 38
		}
	},
	{#State 180
		DEFAULT => -108
	},
	{#State 181
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'equality_expression' => 22,
			'conditional_expression' => 203,
			'shift_expression' => 25,
			'postfix_expression' => 29,
			'additive_expression' => 27,
			'logical_or_expression' => 26,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'unary_operator' => 19,
			'cast_expression' => 38,
			'unary_expression' => 20
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
		DEFAULT => -85
	},
	{#State 187
		ACTIONS => {
			"}" => 206,
			'IDENTIFIER' => 160,
			"{" => 101,
			'CONST' => 54,
			'INLINE' => 53,
			'DO' => 164
		},
		GOTOS => {
			'inline' => 159,
			'spec' => 100,
			'const' => 162,
			'distributed_as' => 155,
			'do' => 156,
			'block' => 163,
			'block_definition' => 190,
			'set_to' => 157
		}
	},
	{#State 188
		ACTIONS => {
			'IDENTIFIER' => 7,
			"{" => 101
		},
		GOTOS => {
			'spec' => 100,
			'block' => 207
		}
	},
	{#State 189
		DEFAULT => -86
	},
	{#State 190
		DEFAULT => -88
	},
	{#State 191
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 208,
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
	{#State 192
		ACTIONS => {
			'IDENTIFIER' => 210
		},
		GOTOS => {
			'dim_aliases' => 209,
			'dim_alias' => 211
		}
	},
	{#State 193
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 212,
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
	{#State 194
		DEFAULT => -82
	},
	{#State 195
		ACTIONS => {
			";" => 213
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
			";" => 214
		},
		DEFAULT => -44
	},
	{#State 199
		DEFAULT => -36
	},
	{#State 200
		DEFAULT => -74
	},
	{#State 201
		DEFAULT => -77
	},
	{#State 202
		DEFAULT => -78
	},
	{#State 203
		DEFAULT => -144
	},
	{#State 204
		ACTIONS => {
			'IDENTIFIER' => 184
		},
		GOTOS => {
			'dim_arg' => 215
		}
	},
	{#State 205
		ACTIONS => {
			"(" => 216
		},
		DEFAULT => -15
	},
	{#State 206
		DEFAULT => -84
	},
	{#State 207
		DEFAULT => -83
	},
	{#State 208
		ACTIONS => {
			";" => 217
		},
		DEFAULT => -99
	},
	{#State 209
		ACTIONS => {
			"," => 218,
			"]" => 219
		}
	},
	{#State 210
		DEFAULT => -73
	},
	{#State 211
		DEFAULT => -72
	},
	{#State 212
		ACTIONS => {
			";" => 220
		},
		DEFAULT => -103
	},
	{#State 213
		DEFAULT => -48
	},
	{#State 214
		DEFAULT => -43
	},
	{#State 215
		DEFAULT => -68
	},
	{#State 216
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 24,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14,
			")" => 221
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'positional_args' => 222,
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
			'named_args' => 223,
			'cast_expression' => 38
		}
	},
	{#State 217
		DEFAULT => -98
	},
	{#State 218
		ACTIONS => {
			'IDENTIFIER' => 210
		},
		GOTOS => {
			'dim_alias' => 224
		}
	},
	{#State 219
		ACTIONS => {
			"~" => 225,
			'SET_TO' => 226
		}
	},
	{#State 220
		DEFAULT => -102
	},
	{#State 221
		DEFAULT => -14
	},
	{#State 222
		ACTIONS => {
			"," => 227,
			")" => 228
		}
	},
	{#State 223
		ACTIONS => {
			"," => 89,
			")" => 229
		}
	},
	{#State 224
		DEFAULT => -71
	},
	{#State 225
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 230,
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
	{#State 226
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 64,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
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
			'logical_and_expression' => 34,
			'multiplicative_expression' => 35,
			'cast_expression' => 38
		}
	},
	{#State 227
		ACTIONS => {
			"-" => 11,
			"(" => 33,
			'IDENTIFIER' => 24,
			"!" => 17,
			"+" => 12,
			'STRING_LITERAL' => 36,
			'LITERAL' => 14
		},
		GOTOS => {
			'conditional_expression' => 13,
			'and_expression' => 15,
			'inclusive_or_expression' => 16,
			'expression' => 18,
			'unary_operator' => 19,
			'unary_expression' => 20,
			'equality_expression' => 22,
			'shift_expression' => 25,
			'logical_or_expression' => 26,
			'additive_expression' => 27,
			'postfix_expression' => 29,
			'positional_arg' => 118,
			'named_arg' => 30,
			'exclusive_or_expression' => 31,
			'relational_expression' => 32,
			'multiplicative_expression' => 35,
			'logical_and_expression' => 34,
			'named_args' => 232,
			'cast_expression' => 38
		}
	},
	{#State 228
		DEFAULT => -12
	},
	{#State 229
		DEFAULT => -13
	},
	{#State 230
		ACTIONS => {
			";" => 233
		},
		DEFAULT => -97
	},
	{#State 231
		ACTIONS => {
			";" => 234
		},
		DEFAULT => -101
	},
	{#State 232
		ACTIONS => {
			"," => 89,
			")" => 235
		}
	},
	{#State 233
		DEFAULT => -96
	},
	{#State 234
		DEFAULT => -100
	},
	{#State 235
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
#line 13 "src/bi.yp"
{ $_[0]->model($_[2], $_[4]) }
	],
	[#Rule 5
		 'model', 4,
sub
#line 14 "src/bi.yp"
{ $_[0]->model($_[2]) }
	],
	[#Rule 6
		 'spec', 6,
sub
#line 18 "src/bi.yp"
{ $_[0]->spec($_[1], [], $_[3], $_[5]) }
	],
	[#Rule 7
		 'spec', 4,
sub
#line 19 "src/bi.yp"
{ $_[0]->spec($_[1], [], $_[3]) }
	],
	[#Rule 8
		 'spec', 4,
sub
#line 20 "src/bi.yp"
{ $_[0]->spec($_[1], [], [], $_[3]) }
	],
	[#Rule 9
		 'spec', 3,
sub
#line 21 "src/bi.yp"
{ $_[0]->spec($_[1]) }
	],
	[#Rule 10
		 'spec', 1,
sub
#line 22 "src/bi.yp"
{ $_[0]->spec($_[1]) }
	],
	[#Rule 11
		 'array_spec', 9,
sub
#line 26 "src/bi.yp"
{ $_[0]->spec($_[1], $_[3], $_[6], $_[8]) }
	],
	[#Rule 12
		 'array_spec', 7,
sub
#line 27 "src/bi.yp"
{ $_[0]->spec($_[1], $_[3], $_[6]) }
	],
	[#Rule 13
		 'array_spec', 7,
sub
#line 28 "src/bi.yp"
{ $_[0]->spec($_[1], $_[3], [], $_[6]) }
	],
	[#Rule 14
		 'array_spec', 6,
sub
#line 29 "src/bi.yp"
{ $_[0]->spec($_[1], $_[3]) }
	],
	[#Rule 15
		 'array_spec', 4,
sub
#line 30 "src/bi.yp"
{ $_[0]->spec($_[1], $_[3]) }
	],
	[#Rule 16
		 'positional_args', 3,
sub
#line 34 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 17
		 'positional_args', 1,
sub
#line 35 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 18
		 'positional_arg', 1,
sub
#line 39 "src/bi.yp"
{ $_[0]->positional_arg($_[1]) }
	],
	[#Rule 19
		 'named_args', 3,
sub
#line 43 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 20
		 'named_args', 1,
sub
#line 44 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 21
		 'named_arg', 3,
sub
#line 48 "src/bi.yp"
{ $_[0]->named_arg($_[1], $_[3]) }
	],
	[#Rule 22
		 'model_definitions', 2,
sub
#line 52 "src/bi.yp"
{ $_[0]->append($_[1], $_[2]) }
	],
	[#Rule 23
		 'model_definitions', 1,
sub
#line 53 "src/bi.yp"
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
#line 71 "src/bi.yp"
{ $_[2] }
	],
	[#Rule 36
		 'dim_declarations', 3,
sub
#line 75 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 37
		 'dim_declarations', 1,
sub
#line 76 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 38
		 'dim_declaration', 2,
sub
#line 80 "src/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 39
		 'dim_declaration', 1,
sub
#line 81 "src/bi.yp"
{ $_[0]->dim($_[1]) }
	],
	[#Rule 40
		 'const', 2,
sub
#line 85 "src/bi.yp"
{ $_[2] }
	],
	[#Rule 41
		 'const_declarations', 3,
sub
#line 89 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 42
		 'const_declarations', 1,
sub
#line 90 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 43
		 'const_declaration', 4,
sub
#line 94 "src/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 44
		 'const_declaration', 3,
sub
#line 95 "src/bi.yp"
{ $_[0]->const($_[1], $_[3]) }
	],
	[#Rule 45
		 'inline', 2,
sub
#line 99 "src/bi.yp"
{ $_[2] }
	],
	[#Rule 46
		 'inline_declarations', 3,
sub
#line 103 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 47
		 'inline_declarations', 1,
sub
#line 104 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 48
		 'inline_declaration', 4,
sub
#line 108 "src/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 49
		 'inline_declaration', 3,
sub
#line 109 "src/bi.yp"
{ $_[0]->inline($_[1], $_[3]) }
	],
	[#Rule 50
		 'state', 3,
sub
#line 113 "src/bi.yp"
{ $_[0]->state($_[2]) }
	],
	[#Rule 51
		 'state', 2,
sub
#line 114 "src/bi.yp"
{ $_[0]->state($_[2]) }
	],
	[#Rule 52
		 'state_aux', 3,
sub
#line 118 "src/bi.yp"
{ $_[0]->state_aux($_[2]) }
	],
	[#Rule 53
		 'state_aux', 2,
sub
#line 119 "src/bi.yp"
{ $_[0]->state_aux($_[2]) }
	],
	[#Rule 54
		 'noise', 3,
sub
#line 123 "src/bi.yp"
{ $_[0]->noise($_[2]) }
	],
	[#Rule 55
		 'noise', 2,
sub
#line 124 "src/bi.yp"
{ $_[0]->noise($_[2]) }
	],
	[#Rule 56
		 'input', 3,
sub
#line 128 "src/bi.yp"
{ $_[0]->input($_[2]) }
	],
	[#Rule 57
		 'input', 2,
sub
#line 129 "src/bi.yp"
{ $_[0]->input($_[2]) }
	],
	[#Rule 58
		 'obs', 3,
sub
#line 133 "src/bi.yp"
{ $_[0]->obs($_[2]) }
	],
	[#Rule 59
		 'obs', 2,
sub
#line 134 "src/bi.yp"
{ $_[0]->obs($_[2]) }
	],
	[#Rule 60
		 'param', 3,
sub
#line 138 "src/bi.yp"
{ $_[0]->param($_[2]) }
	],
	[#Rule 61
		 'param', 2,
sub
#line 139 "src/bi.yp"
{ $_[0]->param($_[2]) }
	],
	[#Rule 62
		 'param_aux', 3,
sub
#line 143 "src/bi.yp"
{ $_[0]->param_aux($_[2]) }
	],
	[#Rule 63
		 'param_aux', 2,
sub
#line 144 "src/bi.yp"
{ $_[0]->param_aux($_[2]) }
	],
	[#Rule 64
		 'var_declarations', 3,
sub
#line 148 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 65
		 'var_declarations', 1,
sub
#line 149 "src/bi.yp"
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
#line 158 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 69
		 'dim_args', 1,
sub
#line 159 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 70
		 'dim_arg', 1,
sub
#line 163 "src/bi.yp"
{ $_[0]->dim_arg($_[1]) }
	],
	[#Rule 71
		 'dim_aliases', 3,
sub
#line 167 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 72
		 'dim_aliases', 1,
sub
#line 168 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 73
		 'dim_alias', 1,
sub
#line 172 "src/bi.yp"
{ $_[0]->dim_alias($_[1]) }
	],
	[#Rule 74
		 'offset_args', 3,
sub
#line 176 "src/bi.yp"
{ $_[0]->append($_[1], $_[3]) }
	],
	[#Rule 75
		 'offset_args', 1,
sub
#line 177 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 76
		 'offset_arg', 1,
sub
#line 181 "src/bi.yp"
{ $_[0]->offset($_[1]) }
	],
	[#Rule 77
		 'offset_arg', 3,
sub
#line 182 "src/bi.yp"
{ $_[0]->offset($_[1], -1, $_[3]) }
	],
	[#Rule 78
		 'offset_arg', 3,
sub
#line 183 "src/bi.yp"
{ $_[0]->offset($_[1], 1, $_[3]) }
	],
	[#Rule 79
		 'offset_arg', 2,
sub
#line 184 "src/bi.yp"
{ $_[0]->offset($_[1], -1, $_[2]) }
	],
	[#Rule 80
		 'offset_arg', 1,
sub
#line 185 "src/bi.yp"
{ $_[0]->offset($_[1], 1, $_[2]) }
	],
	[#Rule 81
		 'top_block', 2,
sub
#line 189 "src/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 82
		 'do', 2,
sub
#line 193 "src/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 83
		 'do', 3,
sub
#line 194 "src/bi.yp"
{ $_[0]->append($_[1], $_[0]->commit_block($_[3])) }
	],
	[#Rule 84
		 'block', 4,
sub
#line 198 "src/bi.yp"
{ $_[0]->block($_[1], $_[3]) }
	],
	[#Rule 85
		 'block', 3,
sub
#line 199 "src/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 86
		 'block', 3,
sub
#line 200 "src/bi.yp"
{ $_[0]->block(undef, $_[2]) }
	],
	[#Rule 87
		 'block', 2,
sub
#line 201 "src/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 88
		 'block_definitions', 2,
sub
#line 205 "src/bi.yp"
{ $_[0]->append($_[1], $_[2]) }
	],
	[#Rule 89
		 'block_definitions', 1,
sub
#line 206 "src/bi.yp"
{ $_[0]->append($_[1]) }
	],
	[#Rule 90
		 'block_definition', 1, undef
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
		 'distributed_as', 7,
sub
#line 219 "src/bi.yp"
{ $_[0]->action($_[1], $_[3], $_[5], $_[6]) }
	],
	[#Rule 97
		 'distributed_as', 6,
sub
#line 220 "src/bi.yp"
{ $_[0]->action($_[1], $_[3], $_[5], $_[6]) }
	],
	[#Rule 98
		 'distributed_as', 4,
sub
#line 221 "src/bi.yp"
{ $_[0]->action($_[1], undef, $_[2], $_[3]) }
	],
	[#Rule 99
		 'distributed_as', 3,
sub
#line 222 "src/bi.yp"
{ $_[0]->action($_[1], undef, $_[2], $_[3]) }
	],
	[#Rule 100
		 'set_to', 7,
sub
#line 226 "src/bi.yp"
{ $_[0]->action($_[1], $_[3], $_[5], $_[6]) }
	],
	[#Rule 101
		 'set_to', 6,
sub
#line 227 "src/bi.yp"
{ $_[0]->action($_[1], $_[3], $_[5], $_[6]) }
	],
	[#Rule 102
		 'set_to', 4,
sub
#line 228 "src/bi.yp"
{ $_[0]->action($_[1], undef, $_[2], $_[3]) }
	],
	[#Rule 103
		 'set_to', 3,
sub
#line 229 "src/bi.yp"
{ $_[0]->action($_[1], undef, $_[2], $_[3]) }
	],
	[#Rule 104
		 'postfix_expression', 1,
sub
#line 238 "src/bi.yp"
{ $_[0]->literal($_[1]) }
	],
	[#Rule 105
		 'postfix_expression', 1,
sub
#line 239 "src/bi.yp"
{ $_[0]->string_literal($_[1]) }
	],
	[#Rule 106
		 'postfix_expression', 1,
sub
#line 240 "src/bi.yp"
{ $_[0]->identifier($_[1]) }
	],
	[#Rule 107
		 'postfix_expression', 4,
sub
#line 241 "src/bi.yp"
{ $_[0]->identifier($_[1], $_[3]) }
	],
	[#Rule 108
		 'postfix_expression', 4,
sub
#line 242 "src/bi.yp"
{ $_[0]->function($_[1], $_[3]) }
	],
	[#Rule 109
		 'postfix_expression', 3,
sub
#line 243 "src/bi.yp"
{ $_[0]->function($_[1]) }
	],
	[#Rule 110
		 'postfix_expression', 3,
sub
#line 244 "src/bi.yp"
{ $_[0]->parens($_[2]) }
	],
	[#Rule 111
		 'unary_expression', 1,
sub
#line 248 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 112
		 'unary_expression', 2,
sub
#line 249 "src/bi.yp"
{ $_[0]->unary_operator($_[1], $_[2]) }
	],
	[#Rule 113
		 'unary_operator', 1,
sub
#line 255 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 114
		 'unary_operator', 1,
sub
#line 256 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 115
		 'unary_operator', 1,
sub
#line 258 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 116
		 'cast_expression', 1,
sub
#line 262 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 117
		 'multiplicative_expression', 1,
sub
#line 267 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 118
		 'multiplicative_expression', 3,
sub
#line 268 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 119
		 'multiplicative_expression', 3,
sub
#line 269 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 120
		 'multiplicative_expression', 3,
sub
#line 270 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 121
		 'multiplicative_expression', 3,
sub
#line 271 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 122
		 'additive_expression', 1,
sub
#line 276 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 123
		 'additive_expression', 3,
sub
#line 277 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 124
		 'additive_expression', 3,
sub
#line 278 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 125
		 'additive_expression', 3,
sub
#line 279 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 126
		 'additive_expression', 3,
sub
#line 280 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 127
		 'shift_expression', 1,
sub
#line 284 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 128
		 'relational_expression', 1,
sub
#line 290 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 129
		 'relational_expression', 3,
sub
#line 291 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 130
		 'relational_expression', 3,
sub
#line 292 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 131
		 'relational_expression', 3,
sub
#line 293 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 132
		 'relational_expression', 3,
sub
#line 294 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 133
		 'equality_expression', 1,
sub
#line 298 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 134
		 'equality_expression', 3,
sub
#line 299 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 135
		 'equality_expression', 3,
sub
#line 300 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 136
		 'and_expression', 1,
sub
#line 304 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 137
		 'exclusive_or_expression', 1,
sub
#line 309 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 138
		 'inclusive_or_expression', 1,
sub
#line 314 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 139
		 'logical_and_expression', 1,
sub
#line 319 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 140
		 'logical_and_expression', 3,
sub
#line 320 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 141
		 'logical_or_expression', 1,
sub
#line 324 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 142
		 'logical_or_expression', 3,
sub
#line 325 "src/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 143
		 'conditional_expression', 1,
sub
#line 329 "src/bi.yp"
{ $_[1] }
	],
	[#Rule 144
		 'conditional_expression', 5,
sub
#line 330 "src/bi.yp"
{ $_[0]->ternary_operator($_[1], $_[2], $_[3], $_[4], $_[5]) }
	],
	[#Rule 145
		 'expression', 1,
sub
#line 334 "src/bi.yp"
{ $_[0]->expression($_[1]) }
	]
],
                                  @_);
    bless($self,$class);
}

#line 337 "src/bi.yp"


1;

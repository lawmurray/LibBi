####################################################################
#
#    This file was generated using Parse::Yapp version 1.21.
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

    my($self)=$class->SUPER::new( yyversion => '1.21',
                                  yystates =>
[
	{#State 0
		ACTIONS => {
			'MODEL' => 2
		},
		GOTOS => {
			'model' => 1
		}
	},
	{#State 1
		ACTIONS => {
			'' => 3
		}
	},
	{#State 2
		ACTIONS => {
			'IDENTIFIER' => 5
		},
		GOTOS => {
			'spec' => 4
		}
	},
	{#State 3
		DEFAULT => 0
	},
	{#State 4
		ACTIONS => {
			"{" => 6
		},
		GOTOS => {
			'model_start' => 7
		}
	},
	{#State 5
		ACTIONS => {
			"(" => 8
		},
		DEFAULT => -9
	},
	{#State 6
		DEFAULT => -3
	},
	{#State 7
		ACTIONS => {
			'OBS' => 12,
			'INLINE' => 13,
			'CONST' => 15,
			'PARAM' => 29,
			'NOISE' => 16,
			'PARAM_AUX' => 31,
			'STATE' => 32,
			'SUB' => 21,
			"}" => 34,
			'DIM' => 22,
			'STATE_AUX' => 27,
			'INPUT' => 26
		},
		GOTOS => {
			'top_level' => 20,
			'dim' => 18,
			'inline' => 19,
			'model_definitions' => 25,
			'model_end' => 35,
			'input' => 23,
			'state' => 33,
			'noise' => 24,
			'state_aux' => 14,
			'obs' => 10,
			'param' => 9,
			'model_definition' => 28,
			'param_aux' => 11,
			'const' => 17,
			'var' => 30
		}
	},
	{#State 8
		ACTIONS => {
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 40,
			"(" => 38,
			")" => 56,
			"+" => 39,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'named_args' => 65,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'postfix_expression' => 58,
			'expression' => 55,
			'additive_expression' => 54,
			'and_expression' => 57,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'cast_expression' => 53,
			'positional_arg' => 46,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'named_arg' => 42,
			'positional_args' => 44,
			'logical_or_expression' => 43,
			'conditional_expression' => 36,
			'equality_expression' => 37
		}
	},
	{#State 9
		DEFAULT => -49
	},
	{#State 10
		DEFAULT => -48
	},
	{#State 11
		DEFAULT => -50
	},
	{#State 12
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'array_spec' => 68,
			'spec' => 70,
			'obs_declarations' => 66,
			'obs_declaration' => 67
		}
	},
	{#State 13
		ACTIONS => {
			'IDENTIFIER' => 73
		},
		GOTOS => {
			'inline_declarations' => 71,
			'inline_declaration' => 72
		}
	},
	{#State 14
		DEFAULT => -45
	},
	{#State 15
		ACTIONS => {
			'IDENTIFIER' => 76
		},
		GOTOS => {
			'const_declaration' => 75,
			'const_declarations' => 74
		}
	},
	{#State 16
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'noise_declaration' => 77,
			'noise_declarations' => 79,
			'array_spec' => 78,
			'spec' => 70
		}
	},
	{#State 17
		DEFAULT => -26
	},
	{#State 18
		DEFAULT => -24
	},
	{#State 19
		DEFAULT => -27
	},
	{#State 20
		DEFAULT => -28
	},
	{#State 21
		ACTIONS => {
			"{" => 82,
			'IDENTIFIER' => 5
		},
		GOTOS => {
			'spec' => 80,
			'block' => 81,
			'block_start' => 83
		}
	},
	{#State 22
		ACTIONS => {
			'IDENTIFIER' => 5
		},
		GOTOS => {
			'dim_declaration' => 85,
			'spec' => 86,
			'dim_declarations' => 84
		}
	},
	{#State 23
		DEFAULT => -47
	},
	{#State 24
		DEFAULT => -46
	},
	{#State 25
		ACTIONS => {
			'INLINE' => 13,
			'OBS' => 12,
			'CONST' => 15,
			'PARAM' => 29,
			'NOISE' => 16,
			'PARAM_AUX' => 31,
			'STATE' => 32,
			'SUB' => 21,
			"}" => 34,
			'DIM' => 22,
			'STATE_AUX' => 27,
			'INPUT' => 26
		},
		GOTOS => {
			'const' => 17,
			'var' => 30,
			'state_aux' => 14,
			'param' => 9,
			'obs' => 10,
			'model_definition' => 88,
			'param_aux' => 11,
			'model_end' => 87,
			'state' => 33,
			'noise' => 24,
			'input' => 23,
			'top_level' => 20,
			'dim' => 18,
			'inline' => 19
		}
	},
	{#State 26
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'array_spec' => 90,
			'spec' => 70,
			'input_declarations' => 89,
			'input_declaration' => 91
		}
	},
	{#State 27
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'state_aux_declaration' => 94,
			'array_spec' => 92,
			'spec' => 70,
			'state_aux_declarations' => 93
		}
	},
	{#State 28
		DEFAULT => -23
	},
	{#State 29
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'spec' => 70,
			'array_spec' => 97,
			'param_declaration' => 95,
			'param_declarations' => 96
		}
	},
	{#State 30
		DEFAULT => -25
	},
	{#State 31
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'param_aux_declarations' => 99,
			'array_spec' => 98,
			'param_aux_declaration' => 100,
			'spec' => 70
		}
	},
	{#State 32
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'spec' => 70,
			'array_spec' => 101,
			'state_declarations' => 103,
			'state_declaration' => 102
		}
	},
	{#State 33
		DEFAULT => -44
	},
	{#State 34
		DEFAULT => -4
	},
	{#State 35
		DEFAULT => -2
	},
	{#State 36
		DEFAULT => -181
	},
	{#State 37
		ACTIONS => {
			'EQ_OP' => 104,
			'NE_OP' => 105
		},
		DEFAULT => -172
	},
	{#State 38
		ACTIONS => {
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'and_expression' => 57,
			'additive_expression' => 54,
			'postfix_expression' => 58,
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'equality_expression' => 37,
			'conditional_expression' => 107,
			'logical_or_expression' => 43,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52
		}
	},
	{#State 39
		DEFAULT => -144
	},
	{#State 40
		ACTIONS => {
			"=" => 109,
			"(" => 110,
			"[" => 108
		},
		DEFAULT => -135
	},
	{#State 41
		DEFAULT => -133
	},
	{#State 42
		DEFAULT => -20
	},
	{#State 43
		ACTIONS => {
			'OR_OP' => 112,
			"?" => 111
		},
		DEFAULT => -179
	},
	{#State 44
		ACTIONS => {
			"," => 114,
			")" => 113
		}
	},
	{#State 45
		ACTIONS => {
			'ELEM_POW_OP' => 115,
			'POW_OP' => 116
		},
		DEFAULT => -151
	},
	{#State 46
		DEFAULT => -17
	},
	{#State 47
		ACTIONS => {
			'AND_OP' => 117
		},
		DEFAULT => -177
	},
	{#State 48
		ACTIONS => {
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'cast_expression' => 118,
			'postfix_expression' => 58,
			'unary_expression' => 49,
			'unary_operator' => 48
		}
	},
	{#State 49
		DEFAULT => -147
	},
	{#State 50
		DEFAULT => -175
	},
	{#State 51
		DEFAULT => -174
	},
	{#State 52
		ACTIONS => {
			"<" => 122,
			'LE_OP' => 121,
			">" => 119,
			'GE_OP' => 120
		},
		DEFAULT => -169
	},
	{#State 53
		DEFAULT => -148
	},
	{#State 54
		ACTIONS => {
			"+" => 126,
			'ELEM_ADD_OP' => 125,
			'ELEM_SUB_OP' => 123,
			"-" => 124
		},
		DEFAULT => -163
	},
	{#State 55
		DEFAULT => -18
	},
	{#State 56
		DEFAULT => -8
	},
	{#State 57
		DEFAULT => -173
	},
	{#State 58
		DEFAULT => -142
	},
	{#State 59
		DEFAULT => -145
	},
	{#State 60
		DEFAULT => -132
	},
	{#State 61
		ACTIONS => {
			"." => 127,
			"/" => 128,
			"*" => 129,
			'ELEM_MUL_OP' => 130,
			"%" => 131,
			'ELEM_DIV_OP' => 132
		},
		DEFAULT => -158
	},
	{#State 62
		DEFAULT => -134
	},
	{#State 63
		DEFAULT => -146
	},
	{#State 64
		DEFAULT => -164
	},
	{#State 65
		ACTIONS => {
			"," => 133,
			")" => 134
		}
	},
	{#State 66
		ACTIONS => {
			"," => 135
		},
		DEFAULT => -71
	},
	{#State 67
		DEFAULT => -73
	},
	{#State 68
		ACTIONS => {
			";" => 136
		},
		DEFAULT => -75
	},
	{#State 69
		ACTIONS => {
			"[" => 137,
			"(" => 8
		},
		DEFAULT => -9
	},
	{#State 70
		DEFAULT => -10
	},
	{#State 71
		ACTIONS => {
			"," => 138
		},
		DEFAULT => -39
	},
	{#State 72
		DEFAULT => -41
	},
	{#State 73
		ACTIONS => {
			"=" => 139
		}
	},
	{#State 74
		ACTIONS => {
			"," => 140
		},
		DEFAULT => -34
	},
	{#State 75
		DEFAULT => -36
	},
	{#State 76
		ACTIONS => {
			"=" => 141
		}
	},
	{#State 77
		DEFAULT => -63
	},
	{#State 78
		ACTIONS => {
			";" => 142
		},
		DEFAULT => -65
	},
	{#State 79
		ACTIONS => {
			"," => 143
		},
		DEFAULT => -61
	},
	{#State 80
		ACTIONS => {
			"{" => 82
		},
		GOTOS => {
			'block_start' => 144
		}
	},
	{#State 81
		DEFAULT => -101
	},
	{#State 82
		DEFAULT => -108
	},
	{#State 83
		ACTIONS => {
			'STATE' => 32,
			'IDENTIFIER' => 161,
			"}" => 147,
			"{" => 82,
			'DIM' => 22,
			'INPUT' => 26,
			'STATE_AUX' => 27,
			'INLINE' => 13,
			'OBS' => 12,
			'CONST' => 15,
			'PARAM' => 29,
			'NOISE' => 16,
			'PARAM_AUX' => 31,
			'DO' => 145
		},
		GOTOS => {
			'noise' => 24,
			'input' => 23,
			'dim' => 157,
			'spec' => 80,
			'inline' => 162,
			'block_definitions' => 160,
			'distributed_as' => 151,
			'block_definition' => 154,
			'do' => 153,
			'block_end' => 152,
			'const' => 150,
			'obs' => 10,
			'param' => 9,
			'target' => 155,
			'param_aux' => 11,
			'block' => 156,
			'state_aux' => 14,
			'action' => 159,
			'state' => 33,
			'set_to' => 148,
			'dtarget' => 149,
			'var' => 146,
			'varies_as' => 158,
			'block_start' => 83
		}
	},
	{#State 84
		ACTIONS => {
			"," => 163
		},
		DEFAULT => -29
	},
	{#State 85
		DEFAULT => -31
	},
	{#State 86
		ACTIONS => {
			";" => 164
		},
		DEFAULT => -33
	},
	{#State 87
		DEFAULT => -1
	},
	{#State 88
		DEFAULT => -22
	},
	{#State 89
		ACTIONS => {
			"," => 165
		},
		DEFAULT => -66
	},
	{#State 90
		ACTIONS => {
			";" => 166
		},
		DEFAULT => -70
	},
	{#State 91
		DEFAULT => -68
	},
	{#State 92
		ACTIONS => {
			";" => 167
		},
		DEFAULT => -60
	},
	{#State 93
		ACTIONS => {
			"," => 168
		},
		DEFAULT => -56
	},
	{#State 94
		DEFAULT => -58
	},
	{#State 95
		DEFAULT => -78
	},
	{#State 96
		ACTIONS => {
			"," => 169
		},
		DEFAULT => -76
	},
	{#State 97
		ACTIONS => {
			";" => 170
		},
		DEFAULT => -80
	},
	{#State 98
		ACTIONS => {
			";" => 171
		},
		DEFAULT => -85
	},
	{#State 99
		ACTIONS => {
			"," => 172
		},
		DEFAULT => -81
	},
	{#State 100
		DEFAULT => -83
	},
	{#State 101
		ACTIONS => {
			";" => 173
		},
		DEFAULT => -55
	},
	{#State 102
		DEFAULT => -53
	},
	{#State 103
		ACTIONS => {
			"," => 174
		},
		DEFAULT => -51
	},
	{#State 104
		ACTIONS => {
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'postfix_expression' => 58,
			'additive_expression' => 54,
			'relational_expression' => 175,
			'cast_expression' => 53,
			'shift_expression' => 64,
			'pow_expression' => 45,
			'unary_operator' => 48,
			'multiplicative_expression' => 61,
			'unary_expression' => 49
		}
	},
	{#State 105
		ACTIONS => {
			"+" => 39,
			"(" => 38,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			'LITERAL' => 60,
			'STRING_LITERAL' => 62,
			"!" => 63
		},
		GOTOS => {
			'shift_expression' => 64,
			'cast_expression' => 53,
			'relational_expression' => 176,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'multiplicative_expression' => 61,
			'pow_expression' => 45,
			'postfix_expression' => 58,
			'additive_expression' => 54
		}
	},
	{#State 106
		ACTIONS => {
			"[" => 108,
			"(" => 110
		},
		DEFAULT => -135
	},
	{#State 107
		ACTIONS => {
			")" => 177
		}
	},
	{#State 108
		ACTIONS => {
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59,
			"+" => 39,
			"(" => 38,
			":" => 181,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'postfix_expression' => 58,
			'additive_expression' => 54,
			'expression' => 180,
			'and_expression' => 57,
			'shift_expression' => 64,
			'index_arg' => 178,
			'multiplicative_expression' => 61,
			'logical_or_expression' => 43,
			'conditional_expression' => 36,
			'index_args' => 179,
			'equality_expression' => 37,
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'cast_expression' => 53,
			'pow_expression' => 45,
			'logical_and_expression' => 47,
			'unary_operator' => 48,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49
		}
	},
	{#State 109
		ACTIONS => {
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			"+" => 39,
			"(" => 38
		},
		GOTOS => {
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'additive_expression' => 54,
			'expression' => 182,
			'and_expression' => 57,
			'postfix_expression' => 58,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'unary_operator' => 48,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'cast_expression' => 53,
			'conditional_expression' => 36,
			'equality_expression' => 37,
			'logical_or_expression' => 43
		}
	},
	{#State 110
		ACTIONS => {
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 40,
			"+" => 39,
			"(" => 38,
			")" => 185,
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60
		},
		GOTOS => {
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'named_args' => 184,
			'and_expression' => 57,
			'expression' => 55,
			'additive_expression' => 54,
			'postfix_expression' => 58,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'positional_arg' => 46,
			'pow_expression' => 45,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'equality_expression' => 37,
			'conditional_expression' => 36,
			'named_arg' => 42,
			'positional_args' => 183,
			'logical_or_expression' => 43
		}
	},
	{#State 111
		ACTIONS => {
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'postfix_expression' => 58,
			'and_expression' => 57,
			'additive_expression' => 54,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'logical_or_expression' => 43,
			'equality_expression' => 37,
			'conditional_expression' => 186,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'pow_expression' => 45,
			'logical_and_expression' => 47
		}
	},
	{#State 112
		ACTIONS => {
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59,
			"+" => 39,
			"(" => 38,
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60
		},
		GOTOS => {
			'postfix_expression' => 58,
			'equality_expression' => 37,
			'and_expression' => 57,
			'additive_expression' => 54,
			'shift_expression' => 64,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'multiplicative_expression' => 61,
			'pow_expression' => 45,
			'logical_and_expression' => 187
		}
	},
	{#State 113
		DEFAULT => -6
	},
	{#State 114
		ACTIONS => {
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60,
			"-" => 59,
			'IDENTIFIER' => 40,
			'INTEGER_LITERAL' => 41,
			"+" => 39,
			"(" => 38
		},
		GOTOS => {
			'additive_expression' => 54,
			'expression' => 55,
			'and_expression' => 57,
			'postfix_expression' => 58,
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'named_args' => 189,
			'conditional_expression' => 36,
			'equality_expression' => 37,
			'logical_or_expression' => 43,
			'named_arg' => 42,
			'pow_expression' => 45,
			'logical_and_expression' => 47,
			'positional_arg' => 188,
			'unary_operator' => 48,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'cast_expression' => 53
		}
	},
	{#State 115
		ACTIONS => {
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59,
			"(" => 38,
			"+" => 39,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'cast_expression' => 190,
			'postfix_expression' => 58,
			'unary_operator' => 48,
			'unary_expression' => 49
		}
	},
	{#State 116
		ACTIONS => {
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62,
			"(" => 38,
			"+" => 39,
			"-" => 59,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41
		},
		GOTOS => {
			'unary_expression' => 49,
			'unary_operator' => 48,
			'cast_expression' => 191,
			'postfix_expression' => 58
		}
	},
	{#State 117
		ACTIONS => {
			'LITERAL' => 60,
			'STRING_LITERAL' => 62,
			"!" => 63,
			"+" => 39,
			"(" => 38,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59
		},
		GOTOS => {
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'cast_expression' => 53,
			'shift_expression' => 64,
			'pow_expression' => 45,
			'unary_operator' => 48,
			'multiplicative_expression' => 61,
			'inclusive_or_expression' => 192,
			'unary_expression' => 49,
			'postfix_expression' => 58,
			'additive_expression' => 54,
			'equality_expression' => 37,
			'and_expression' => 57
		}
	},
	{#State 118
		DEFAULT => -143
	},
	{#State 119
		ACTIONS => {
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"+" => 39,
			"(" => 38,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'postfix_expression' => 58,
			'additive_expression' => 54,
			'shift_expression' => 193,
			'cast_expression' => 53,
			'pow_expression' => 45,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'multiplicative_expression' => 61
		}
	},
	{#State 120
		ACTIONS => {
			"+" => 39,
			"(" => 38,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59,
			'LITERAL' => 60,
			'STRING_LITERAL' => 62,
			"!" => 63
		},
		GOTOS => {
			'pow_expression' => 45,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'multiplicative_expression' => 61,
			'shift_expression' => 194,
			'cast_expression' => 53,
			'additive_expression' => 54,
			'postfix_expression' => 58
		}
	},
	{#State 121
		ACTIONS => {
			"-" => 59,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"+" => 39,
			"(" => 38,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'shift_expression' => 195,
			'cast_expression' => 53,
			'multiplicative_expression' => 61,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'pow_expression' => 45,
			'postfix_expression' => 58,
			'additive_expression' => 54
		}
	},
	{#State 122
		ACTIONS => {
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60,
			"-" => 59,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"+" => 39,
			"(" => 38
		},
		GOTOS => {
			'multiplicative_expression' => 61,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'pow_expression' => 45,
			'shift_expression' => 196,
			'cast_expression' => 53,
			'additive_expression' => 54,
			'postfix_expression' => 58
		}
	},
	{#State 123
		ACTIONS => {
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62,
			"(" => 38,
			"+" => 39,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59
		},
		GOTOS => {
			'unary_expression' => 49,
			'multiplicative_expression' => 197,
			'unary_operator' => 48,
			'pow_expression' => 45,
			'cast_expression' => 53,
			'postfix_expression' => 58
		}
	},
	{#State 124
		ACTIONS => {
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			"(" => 38,
			"+" => 39,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'cast_expression' => 53,
			'postfix_expression' => 58,
			'unary_expression' => 49,
			'multiplicative_expression' => 198,
			'unary_operator' => 48,
			'pow_expression' => 45
		}
	},
	{#State 125
		ACTIONS => {
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"+" => 39,
			"(" => 38
		},
		GOTOS => {
			'unary_expression' => 49,
			'multiplicative_expression' => 199,
			'unary_operator' => 48,
			'pow_expression' => 45,
			'cast_expression' => 53,
			'postfix_expression' => 58
		}
	},
	{#State 126
		ACTIONS => {
			"+" => 39,
			"(" => 38,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62
		},
		GOTOS => {
			'unary_expression' => 49,
			'multiplicative_expression' => 200,
			'unary_operator' => 48,
			'pow_expression' => 45,
			'cast_expression' => 53,
			'postfix_expression' => 58
		}
	},
	{#State 127
		ACTIONS => {
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			"+" => 39,
			"(" => 38,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'unary_expression' => 49,
			'unary_operator' => 48,
			'pow_expression' => 201,
			'cast_expression' => 53,
			'postfix_expression' => 58
		}
	},
	{#State 128
		ACTIONS => {
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60,
			"-" => 59,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'cast_expression' => 53,
			'postfix_expression' => 58,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'pow_expression' => 202
		}
	},
	{#State 129
		ACTIONS => {
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'postfix_expression' => 58,
			'cast_expression' => 53,
			'pow_expression' => 203,
			'unary_expression' => 49,
			'unary_operator' => 48
		}
	},
	{#State 130
		ACTIONS => {
			"-" => 59,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"+" => 39,
			"(" => 38,
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60
		},
		GOTOS => {
			'postfix_expression' => 58,
			'cast_expression' => 53,
			'pow_expression' => 204,
			'unary_expression' => 49,
			'unary_operator' => 48
		}
	},
	{#State 131
		ACTIONS => {
			"(" => 38,
			"+" => 39,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62
		},
		GOTOS => {
			'unary_expression' => 49,
			'unary_operator' => 48,
			'pow_expression' => 205,
			'cast_expression' => 53,
			'postfix_expression' => 58
		}
	},
	{#State 132
		ACTIONS => {
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'unary_expression' => 49,
			'unary_operator' => 48,
			'pow_expression' => 206,
			'cast_expression' => 53,
			'postfix_expression' => 58
		}
	},
	{#State 133
		ACTIONS => {
			'IDENTIFIER' => 208
		},
		GOTOS => {
			'named_arg' => 207
		}
	},
	{#State 134
		DEFAULT => -7
	},
	{#State 135
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'array_spec' => 68,
			'spec' => 70,
			'obs_declaration' => 209
		}
	},
	{#State 136
		DEFAULT => -74
	},
	{#State 137
		ACTIONS => {
			'IDENTIFIER' => 212
		},
		GOTOS => {
			'dim_arg' => 210,
			'dim_args' => 211
		}
	},
	{#State 138
		ACTIONS => {
			'IDENTIFIER' => 73
		},
		GOTOS => {
			'inline_declaration' => 213
		}
	},
	{#State 139
		ACTIONS => {
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"(" => 38,
			"+" => 39
		},
		GOTOS => {
			'postfix_expression' => 58,
			'additive_expression' => 54,
			'expression' => 214,
			'and_expression' => 57,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'logical_or_expression' => 43,
			'conditional_expression' => 36,
			'equality_expression' => 37,
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'cast_expression' => 53,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48
		}
	},
	{#State 140
		ACTIONS => {
			'IDENTIFIER' => 76
		},
		GOTOS => {
			'const_declaration' => 215
		}
	},
	{#State 141
		ACTIONS => {
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59,
			"+" => 39,
			"(" => 38,
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60
		},
		GOTOS => {
			'logical_or_expression' => 43,
			'conditional_expression' => 36,
			'equality_expression' => 37,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'cast_expression' => 53,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'unary_operator' => 48,
			'postfix_expression' => 58,
			'additive_expression' => 54,
			'expression' => 216,
			'and_expression' => 57,
			'shift_expression' => 64,
			'multiplicative_expression' => 61
		}
	},
	{#State 142
		DEFAULT => -64
	},
	{#State 143
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'spec' => 70,
			'array_spec' => 78,
			'noise_declaration' => 217
		}
	},
	{#State 144
		ACTIONS => {
			"}" => 147,
			"{" => 82,
			'DIM' => 22,
			'INPUT' => 26,
			'STATE_AUX' => 27,
			'STATE' => 32,
			'IDENTIFIER' => 161,
			'NOISE' => 16,
			'DO' => 145,
			'PARAM_AUX' => 31,
			'INLINE' => 13,
			'OBS' => 12,
			'CONST' => 15,
			'PARAM' => 29
		},
		GOTOS => {
			'state' => 33,
			'dtarget' => 149,
			'set_to' => 148,
			'var' => 146,
			'varies_as' => 158,
			'block_start' => 83,
			'input' => 23,
			'noise' => 24,
			'inline' => 162,
			'dim' => 157,
			'spec' => 80,
			'block_definitions' => 219,
			'do' => 153,
			'block_definition' => 154,
			'block_end' => 218,
			'distributed_as' => 151,
			'const' => 150,
			'param_aux' => 11,
			'block' => 156,
			'param' => 9,
			'obs' => 10,
			'target' => 155,
			'action' => 159,
			'state_aux' => 14
		}
	},
	{#State 145
		ACTIONS => {
			'IDENTIFIER' => 5,
			"{" => 82
		},
		GOTOS => {
			'block' => 220,
			'spec' => 80,
			'block_start' => 83
		}
	},
	{#State 146
		DEFAULT => -116
	},
	{#State 147
		DEFAULT => -109
	},
	{#State 148
		DEFAULT => -120
	},
	{#State 149
		ACTIONS => {
			"=" => 221
		}
	},
	{#State 150
		DEFAULT => -113
	},
	{#State 151
		DEFAULT => -119
	},
	{#State 152
		DEFAULT => -107
	},
	{#State 153
		ACTIONS => {
			'THEN' => 222
		},
		DEFAULT => -112
	},
	{#State 154
		DEFAULT => -111
	},
	{#State 155
		ACTIONS => {
			'SET_TO' => 224,
			"~" => 223
		}
	},
	{#State 156
		DEFAULT => -118
	},
	{#State 157
		DEFAULT => -115
	},
	{#State 158
		DEFAULT => -121
	},
	{#State 159
		DEFAULT => -117
	},
	{#State 160
		ACTIONS => {
			'STATE' => 32,
			'IDENTIFIER' => 161,
			'DIM' => 22,
			"{" => 82,
			"}" => 147,
			'STATE_AUX' => 27,
			'INPUT' => 26,
			'OBS' => 12,
			'INLINE' => 13,
			'PARAM' => 29,
			'CONST' => 15,
			'NOISE' => 16,
			'DO' => 145,
			'PARAM_AUX' => 31
		},
		GOTOS => {
			'varies_as' => 158,
			'block_start' => 83,
			'var' => 146,
			'set_to' => 148,
			'dtarget' => 149,
			'state' => 33,
			'target' => 155,
			'obs' => 10,
			'param' => 9,
			'param_aux' => 11,
			'block' => 156,
			'state_aux' => 14,
			'action' => 159,
			'block_end' => 226,
			'do' => 153,
			'block_definition' => 225,
			'distributed_as' => 151,
			'const' => 150,
			'spec' => 80,
			'dim' => 157,
			'inline' => 162,
			'noise' => 24,
			'input' => 23
		}
	},
	{#State 161
		ACTIONS => {
			"[" => 227,
			"{" => -9,
			"/" => 228,
			"(" => 8
		},
		DEFAULT => -129
	},
	{#State 162
		DEFAULT => -114
	},
	{#State 163
		ACTIONS => {
			'IDENTIFIER' => 5
		},
		GOTOS => {
			'dim_declaration' => 229,
			'spec' => 86
		}
	},
	{#State 164
		DEFAULT => -32
	},
	{#State 165
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'input_declaration' => 230,
			'array_spec' => 90,
			'spec' => 70
		}
	},
	{#State 166
		DEFAULT => -69
	},
	{#State 167
		DEFAULT => -59
	},
	{#State 168
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'spec' => 70,
			'array_spec' => 92,
			'state_aux_declaration' => 231
		}
	},
	{#State 169
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'spec' => 70,
			'array_spec' => 97,
			'param_declaration' => 232
		}
	},
	{#State 170
		DEFAULT => -79
	},
	{#State 171
		DEFAULT => -84
	},
	{#State 172
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'array_spec' => 98,
			'spec' => 70,
			'param_aux_declaration' => 233
		}
	},
	{#State 173
		DEFAULT => -54
	},
	{#State 174
		ACTIONS => {
			'IDENTIFIER' => 69
		},
		GOTOS => {
			'array_spec' => 101,
			'spec' => 70,
			'state_declaration' => 234
		}
	},
	{#State 175
		ACTIONS => {
			"<" => 122,
			'LE_OP' => 121,
			'GE_OP' => 120,
			">" => 119
		},
		DEFAULT => -170
	},
	{#State 176
		ACTIONS => {
			"<" => 122,
			'GE_OP' => 120,
			">" => 119,
			'LE_OP' => 121
		},
		DEFAULT => -171
	},
	{#State 177
		DEFAULT => -141
	},
	{#State 178
		DEFAULT => -97
	},
	{#State 179
		ACTIONS => {
			"," => 236,
			"]" => 235
		}
	},
	{#State 180
		ACTIONS => {
			":" => 237
		},
		DEFAULT => -98
	},
	{#State 181
		DEFAULT => -100
	},
	{#State 182
		DEFAULT => -21
	},
	{#State 183
		ACTIONS => {
			")" => 239,
			"," => 238
		}
	},
	{#State 184
		ACTIONS => {
			")" => 240,
			"," => 133
		}
	},
	{#State 185
		DEFAULT => -140
	},
	{#State 186
		ACTIONS => {
			":" => 241
		}
	},
	{#State 187
		ACTIONS => {
			'AND_OP' => 117
		},
		DEFAULT => -178
	},
	{#State 188
		DEFAULT => -16
	},
	{#State 189
		ACTIONS => {
			"," => 133,
			")" => 242
		}
	},
	{#State 190
		DEFAULT => -150
	},
	{#State 191
		DEFAULT => -149
	},
	{#State 192
		DEFAULT => -176
	},
	{#State 193
		DEFAULT => -166
	},
	{#State 194
		DEFAULT => -168
	},
	{#State 195
		DEFAULT => -167
	},
	{#State 196
		DEFAULT => -165
	},
	{#State 197
		ACTIONS => {
			"%" => 131,
			'ELEM_DIV_OP' => 132,
			"*" => 129,
			"/" => 128,
			'ELEM_MUL_OP' => 130,
			"." => 127
		},
		DEFAULT => -162
	},
	{#State 198
		ACTIONS => {
			'ELEM_MUL_OP' => 130,
			"/" => 128,
			"*" => 129,
			'ELEM_DIV_OP' => 132,
			"%" => 131,
			"." => 127
		},
		DEFAULT => -161
	},
	{#State 199
		ACTIONS => {
			'ELEM_MUL_OP' => 130,
			"*" => 129,
			"/" => 128,
			'ELEM_DIV_OP' => 132,
			"%" => 131,
			"." => 127
		},
		DEFAULT => -160
	},
	{#State 200
		ACTIONS => {
			"%" => 131,
			'ELEM_DIV_OP' => 132,
			"*" => 129,
			"/" => 128,
			'ELEM_MUL_OP' => 130,
			"." => 127
		},
		DEFAULT => -159
	},
	{#State 201
		ACTIONS => {
			'ELEM_POW_OP' => 115,
			'POW_OP' => 116
		},
		DEFAULT => -157
	},
	{#State 202
		ACTIONS => {
			'POW_OP' => 116,
			'ELEM_POW_OP' => 115
		},
		DEFAULT => -154
	},
	{#State 203
		ACTIONS => {
			'POW_OP' => 116,
			'ELEM_POW_OP' => 115
		},
		DEFAULT => -152
	},
	{#State 204
		ACTIONS => {
			'ELEM_POW_OP' => 115,
			'POW_OP' => 116
		},
		DEFAULT => -153
	},
	{#State 205
		ACTIONS => {
			'ELEM_POW_OP' => 115,
			'POW_OP' => 116
		},
		DEFAULT => -156
	},
	{#State 206
		ACTIONS => {
			'POW_OP' => 116,
			'ELEM_POW_OP' => 115
		},
		DEFAULT => -155
	},
	{#State 207
		DEFAULT => -19
	},
	{#State 208
		ACTIONS => {
			"=" => 109
		}
	},
	{#State 209
		DEFAULT => -72
	},
	{#State 210
		DEFAULT => -87
	},
	{#State 211
		ACTIONS => {
			"," => 243,
			"]" => 244
		}
	},
	{#State 212
		DEFAULT => -88
	},
	{#State 213
		DEFAULT => -40
	},
	{#State 214
		ACTIONS => {
			";" => 245
		},
		DEFAULT => -43
	},
	{#State 215
		DEFAULT => -35
	},
	{#State 216
		ACTIONS => {
			";" => 246
		},
		DEFAULT => -38
	},
	{#State 217
		DEFAULT => -62
	},
	{#State 218
		DEFAULT => -105
	},
	{#State 219
		ACTIONS => {
			'CONST' => 15,
			'PARAM' => 29,
			'INLINE' => 13,
			'OBS' => 12,
			'PARAM_AUX' => 31,
			'DO' => 145,
			'NOISE' => 16,
			'IDENTIFIER' => 161,
			'STATE' => 32,
			'INPUT' => 26,
			'STATE_AUX' => 27,
			"}" => 147,
			"{" => 82,
			'DIM' => 22
		},
		GOTOS => {
			'block_end' => 247,
			'do' => 153,
			'block_definition' => 225,
			'distributed_as' => 151,
			'const' => 150,
			'block' => 156,
			'param_aux' => 11,
			'target' => 155,
			'obs' => 10,
			'param' => 9,
			'action' => 159,
			'state_aux' => 14,
			'input' => 23,
			'noise' => 24,
			'inline' => 162,
			'dim' => 157,
			'spec' => 80,
			'var' => 146,
			'varies_as' => 158,
			'block_start' => 83,
			'state' => 33,
			'dtarget' => 149,
			'set_to' => 148
		}
	},
	{#State 220
		DEFAULT => -102
	},
	{#State 221
		ACTIONS => {
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			"(" => 38,
			"+" => 39,
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60
		},
		GOTOS => {
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'logical_or_expression' => 43,
			'equality_expression' => 37,
			'conditional_expression' => 36,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'additive_expression' => 54,
			'expression' => 248
		}
	},
	{#State 222
		ACTIONS => {
			'IDENTIFIER' => 5,
			"{" => 82
		},
		GOTOS => {
			'block' => 249,
			'spec' => 80,
			'block_start' => 83
		}
	},
	{#State 223
		ACTIONS => {
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62,
			"(" => 38,
			"+" => 39,
			'IDENTIFIER' => 106,
			'INTEGER_LITERAL' => 41,
			"-" => 59
		},
		GOTOS => {
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'additive_expression' => 54,
			'expression' => 250,
			'and_expression' => 57,
			'postfix_expression' => 58,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'cast_expression' => 53,
			'conditional_expression' => 36,
			'equality_expression' => 37,
			'logical_or_expression' => 43
		}
	},
	{#State 224
		ACTIONS => {
			"(" => 38,
			"+" => 39,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62
		},
		GOTOS => {
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'and_expression' => 57,
			'expression' => 251,
			'additive_expression' => 54,
			'postfix_expression' => 58,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'pow_expression' => 45,
			'logical_and_expression' => 47,
			'cast_expression' => 53,
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'equality_expression' => 37,
			'conditional_expression' => 36,
			'logical_or_expression' => 43
		}
	},
	{#State 225
		DEFAULT => -110
	},
	{#State 226
		DEFAULT => -106
	},
	{#State 227
		ACTIONS => {
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62,
			"+" => 39,
			"(" => 38,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 254
		},
		GOTOS => {
			'dim_aliases' => 253,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'postfix_expression' => 58,
			'dim_alias' => 252,
			'additive_expression' => 54,
			'and_expression' => 57,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'cast_expression' => 53,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'unary_operator' => 48,
			'logical_or_expression' => 255,
			'equality_expression' => 37
		}
	},
	{#State 228
		ACTIONS => {
			'DT' => 256
		}
	},
	{#State 229
		DEFAULT => -30
	},
	{#State 230
		DEFAULT => -67
	},
	{#State 231
		DEFAULT => -57
	},
	{#State 232
		DEFAULT => -77
	},
	{#State 233
		DEFAULT => -82
	},
	{#State 234
		DEFAULT => -52
	},
	{#State 235
		DEFAULT => -136
	},
	{#State 236
		ACTIONS => {
			"+" => 39,
			"(" => 38,
			":" => 181,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62
		},
		GOTOS => {
			'index_arg' => 257,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'expression' => 180,
			'additive_expression' => 54,
			'cast_expression' => 53,
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'logical_or_expression' => 43,
			'equality_expression' => 37,
			'conditional_expression' => 36
		}
	},
	{#State 237
		ACTIONS => {
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"(" => 38,
			"+" => 39,
			"!" => 63,
			'STRING_LITERAL' => 62,
			'LITERAL' => 60
		},
		GOTOS => {
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'unary_operator' => 48,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'pow_expression' => 45,
			'logical_and_expression' => 47,
			'logical_or_expression' => 43,
			'equality_expression' => 37,
			'conditional_expression' => 36,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'expression' => 258,
			'additive_expression' => 54
		}
	},
	{#State 238
		ACTIONS => {
			"(" => 38,
			"+" => 39,
			"-" => 59,
			'IDENTIFIER' => 40,
			'INTEGER_LITERAL' => 41,
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62
		},
		GOTOS => {
			'logical_or_expression' => 43,
			'named_arg' => 42,
			'equality_expression' => 37,
			'conditional_expression' => 36,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'positional_arg' => 188,
			'pow_expression' => 45,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'expression' => 55,
			'additive_expression' => 54,
			'shift_expression' => 64,
			'named_args' => 259,
			'multiplicative_expression' => 61
		}
	},
	{#State 239
		DEFAULT => -138
	},
	{#State 240
		DEFAULT => -139
	},
	{#State 241
		ACTIONS => {
			"(" => 38,
			"+" => 39,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62
		},
		GOTOS => {
			'equality_expression' => 37,
			'conditional_expression' => 260,
			'logical_or_expression' => 43,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'cast_expression' => 53,
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'and_expression' => 57,
			'additive_expression' => 54,
			'postfix_expression' => 58,
			'multiplicative_expression' => 61,
			'shift_expression' => 64
		}
	},
	{#State 242
		DEFAULT => -5
	},
	{#State 243
		ACTIONS => {
			'IDENTIFIER' => 212
		},
		GOTOS => {
			'dim_arg' => 261
		}
	},
	{#State 244
		ACTIONS => {
			"(" => 262
		},
		DEFAULT => -15
	},
	{#State 245
		DEFAULT => -42
	},
	{#State 246
		DEFAULT => -37
	},
	{#State 247
		DEFAULT => -104
	},
	{#State 248
		ACTIONS => {
			";" => 263
		},
		DEFAULT => -127
	},
	{#State 249
		DEFAULT => -103
	},
	{#State 250
		ACTIONS => {
			";" => 264
		},
		DEFAULT => -123
	},
	{#State 251
		ACTIONS => {
			";" => 265
		},
		DEFAULT => -125
	},
	{#State 252
		DEFAULT => -90
	},
	{#State 253
		ACTIONS => {
			"," => 266,
			"]" => 267
		}
	},
	{#State 254
		ACTIONS => {
			"]" => -91,
			"=" => 268,
			"(" => 110,
			"[" => 108,
			"," => -91
		},
		DEFAULT => -135
	},
	{#State 255
		ACTIONS => {
			":" => 269,
			'OR_OP' => 112
		},
		DEFAULT => -94
	},
	{#State 256
		DEFAULT => -131
	},
	{#State 257
		DEFAULT => -96
	},
	{#State 258
		DEFAULT => -99
	},
	{#State 259
		ACTIONS => {
			"," => 133,
			")" => 270
		}
	},
	{#State 260
		DEFAULT => -180
	},
	{#State 261
		DEFAULT => -86
	},
	{#State 262
		ACTIONS => {
			"-" => 59,
			'IDENTIFIER' => 40,
			'INTEGER_LITERAL' => 41,
			")" => 272,
			"+" => 39,
			"(" => 38,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'named_args' => 271,
			'shift_expression' => 64,
			'multiplicative_expression' => 61,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'additive_expression' => 54,
			'expression' => 55,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'pow_expression' => 45,
			'positional_arg' => 46,
			'logical_and_expression' => 47,
			'positional_args' => 273,
			'named_arg' => 42,
			'logical_or_expression' => 43,
			'equality_expression' => 37,
			'conditional_expression' => 36
		}
	},
	{#State 263
		DEFAULT => -126
	},
	{#State 264
		DEFAULT => -122
	},
	{#State 265
		DEFAULT => -124
	},
	{#State 266
		ACTIONS => {
			"-" => 59,
			'IDENTIFIER' => 254,
			'INTEGER_LITERAL' => 41,
			"+" => 39,
			"(" => 38,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'additive_expression' => 54,
			'and_expression' => 57,
			'postfix_expression' => 58,
			'dim_alias' => 274,
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'equality_expression' => 37,
			'logical_or_expression' => 255,
			'pow_expression' => 45,
			'logical_and_expression' => 47,
			'unary_operator' => 48,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'cast_expression' => 53
		}
	},
	{#State 267
		ACTIONS => {
			"/" => 275
		},
		DEFAULT => -128
	},
	{#State 268
		ACTIONS => {
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"(" => 38,
			"+" => 39,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'logical_or_expression' => 276,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'equality_expression' => 37,
			'additive_expression' => 54,
			'shift_expression' => 64,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'multiplicative_expression' => 61,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'pow_expression' => 45
		}
	},
	{#State 269
		ACTIONS => {
			"(" => 38,
			"+" => 39,
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			'LITERAL' => 60,
			'STRING_LITERAL' => 62,
			"!" => 63
		},
		GOTOS => {
			'shift_expression' => 64,
			'cast_expression' => 53,
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'multiplicative_expression' => 61,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'logical_or_expression' => 277,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'equality_expression' => 37,
			'additive_expression' => 54
		}
	},
	{#State 270
		DEFAULT => -137
	},
	{#State 271
		ACTIONS => {
			")" => 278,
			"," => 133
		}
	},
	{#State 272
		DEFAULT => -14
	},
	{#State 273
		ACTIONS => {
			"," => 279,
			")" => 280
		}
	},
	{#State 274
		DEFAULT => -89
	},
	{#State 275
		ACTIONS => {
			'DT' => 281
		}
	},
	{#State 276
		ACTIONS => {
			":" => 282,
			'OR_OP' => 112
		},
		DEFAULT => -92
	},
	{#State 277
		ACTIONS => {
			'OR_OP' => 112
		},
		DEFAULT => -95
	},
	{#State 278
		DEFAULT => -13
	},
	{#State 279
		ACTIONS => {
			"-" => 59,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 40,
			"(" => 38,
			"+" => 39,
			'STRING_LITERAL' => 62,
			"!" => 63,
			'LITERAL' => 60
		},
		GOTOS => {
			'multiplicative_expression' => 61,
			'shift_expression' => 64,
			'named_args' => 283,
			'and_expression' => 57,
			'additive_expression' => 54,
			'expression' => 55,
			'postfix_expression' => 58,
			'unary_operator' => 48,
			'unary_expression' => 49,
			'inclusive_or_expression' => 50,
			'pow_expression' => 45,
			'positional_arg' => 188,
			'logical_and_expression' => 47,
			'cast_expression' => 53,
			'relational_expression' => 52,
			'exclusive_or_expression' => 51,
			'equality_expression' => 37,
			'conditional_expression' => 36,
			'named_arg' => 42,
			'logical_or_expression' => 43
		}
	},
	{#State 280
		DEFAULT => -12
	},
	{#State 281
		DEFAULT => -130
	},
	{#State 282
		ACTIONS => {
			"(" => 38,
			"+" => 39,
			'INTEGER_LITERAL' => 41,
			'IDENTIFIER' => 106,
			"-" => 59,
			'LITERAL' => 60,
			"!" => 63,
			'STRING_LITERAL' => 62
		},
		GOTOS => {
			'shift_expression' => 64,
			'cast_expression' => 53,
			'exclusive_or_expression' => 51,
			'relational_expression' => 52,
			'multiplicative_expression' => 61,
			'inclusive_or_expression' => 50,
			'unary_expression' => 49,
			'unary_operator' => 48,
			'logical_and_expression' => 47,
			'pow_expression' => 45,
			'logical_or_expression' => 284,
			'postfix_expression' => 58,
			'and_expression' => 57,
			'equality_expression' => 37,
			'additive_expression' => 54
		}
	},
	{#State 283
		ACTIONS => {
			")" => 285,
			"," => 133
		}
	},
	{#State 284
		ACTIONS => {
			'OR_OP' => 112
		},
		DEFAULT => -93
	},
	{#State 285
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
		 'index_arg', 1,
sub
#line 244 "share/bi.yp"
{ $_[0]->range() }
	],
	[#Rule 101
		 'top_level', 2,
sub
#line 248 "share/bi.yp"
{ $_[0]->top_level($_[2]) }
	],
	[#Rule 102
		 'do', 2,
sub
#line 252 "share/bi.yp"
{ $_[0]->commit_block($_[2]) }
	],
	[#Rule 103
		 'do', 3,
sub
#line 253 "share/bi.yp"
{ $_[0]->commit_block($_[3]) }
	],
	[#Rule 104
		 'block', 4,
sub
#line 257 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 105
		 'block', 3,
sub
#line 258 "share/bi.yp"
{ $_[0]->block($_[1]) }
	],
	[#Rule 106
		 'block', 3,
sub
#line 259 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 107
		 'block', 2,
sub
#line 260 "share/bi.yp"
{ $_[0]->block() }
	],
	[#Rule 108
		 'block_start', 1,
sub
#line 264 "share/bi.yp"
{ $_[0]->push_block }
	],
	[#Rule 109
		 'block_end', 1, undef
	],
	[#Rule 110
		 'block_definitions', 2, undef
	],
	[#Rule 111
		 'block_definitions', 1, undef
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
		 'block_definition', 1, undef
	],
	[#Rule 119
		 'action', 1, undef
	],
	[#Rule 120
		 'action', 1, undef
	],
	[#Rule 121
		 'action', 1, undef
	],
	[#Rule 122
		 'distributed_as', 4,
sub
#line 293 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 123
		 'distributed_as', 3,
sub
#line 294 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 124
		 'set_to', 4,
sub
#line 298 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 125
		 'set_to', 3,
sub
#line 299 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 126
		 'varies_as', 4,
sub
#line 303 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 127
		 'varies_as', 3,
sub
#line 304 "share/bi.yp"
{ $_[0]->action($_[2], $_[3]) }
	],
	[#Rule 128
		 'target', 4,
sub
#line 308 "share/bi.yp"
{ $_[0]->target($_[1], $_[3]) }
	],
	[#Rule 129
		 'target', 1,
sub
#line 309 "share/bi.yp"
{ $_[0]->target($_[1]) }
	],
	[#Rule 130
		 'dtarget', 6,
sub
#line 313 "share/bi.yp"
{ $_[0]->dtarget($_[1], $_[3]) }
	],
	[#Rule 131
		 'dtarget', 3,
sub
#line 314 "share/bi.yp"
{ $_[0]->dtarget($_[1]) }
	],
	[#Rule 132
		 'postfix_expression', 1,
sub
#line 323 "share/bi.yp"
{ $_[0]->literal($_[1]) }
	],
	[#Rule 133
		 'postfix_expression', 1,
sub
#line 324 "share/bi.yp"
{ $_[0]->integer_literal($_[1]) }
	],
	[#Rule 134
		 'postfix_expression', 1,
sub
#line 325 "share/bi.yp"
{ $_[0]->string_literal($_[1]) }
	],
	[#Rule 135
		 'postfix_expression', 1,
sub
#line 326 "share/bi.yp"
{ $_[0]->identifier($_[1]) }
	],
	[#Rule 136
		 'postfix_expression', 4,
sub
#line 327 "share/bi.yp"
{ $_[0]->identifier($_[1], $_[3]) }
	],
	[#Rule 137
		 'postfix_expression', 6,
sub
#line 328 "share/bi.yp"
{ $_[0]->function($_[1], $_[3], $_[5]) }
	],
	[#Rule 138
		 'postfix_expression', 4,
sub
#line 329 "share/bi.yp"
{ $_[0]->function($_[1], $_[3]) }
	],
	[#Rule 139
		 'postfix_expression', 4,
sub
#line 330 "share/bi.yp"
{ $_[0]->function($_[1], undef, $_[3]) }
	],
	[#Rule 140
		 'postfix_expression', 3,
sub
#line 331 "share/bi.yp"
{ $_[0]->function($_[1]) }
	],
	[#Rule 141
		 'postfix_expression', 3,
sub
#line 332 "share/bi.yp"
{ $_[0]->parens($_[2]) }
	],
	[#Rule 142
		 'unary_expression', 1,
sub
#line 336 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 143
		 'unary_expression', 2,
sub
#line 337 "share/bi.yp"
{ $_[0]->unary_operator($_[1], $_[2]) }
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
#line 344 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 146
		 'unary_operator', 1,
sub
#line 346 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 147
		 'cast_expression', 1,
sub
#line 350 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 148
		 'pow_expression', 1, undef
	],
	[#Rule 149
		 'pow_expression', 3,
sub
#line 356 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 150
		 'pow_expression', 3,
sub
#line 357 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 151
		 'multiplicative_expression', 1,
sub
#line 361 "share/bi.yp"
{ $_[1] }
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
		 'multiplicative_expression', 3,
sub
#line 366 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 157
		 'multiplicative_expression', 3,
sub
#line 367 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 158
		 'additive_expression', 1,
sub
#line 371 "share/bi.yp"
{ $_[1] }
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
		 'additive_expression', 3,
sub
#line 374 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 162
		 'additive_expression', 3,
sub
#line 375 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 163
		 'shift_expression', 1,
sub
#line 379 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 164
		 'relational_expression', 1,
sub
#line 385 "share/bi.yp"
{ $_[1] }
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
		 'relational_expression', 3,
sub
#line 388 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 168
		 'relational_expression', 3,
sub
#line 389 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 169
		 'equality_expression', 1,
sub
#line 393 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 170
		 'equality_expression', 3,
sub
#line 394 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 171
		 'equality_expression', 3,
sub
#line 395 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 172
		 'and_expression', 1,
sub
#line 399 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 173
		 'exclusive_or_expression', 1,
sub
#line 404 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 174
		 'inclusive_or_expression', 1,
sub
#line 409 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 175
		 'logical_and_expression', 1,
sub
#line 414 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 176
		 'logical_and_expression', 3,
sub
#line 415 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 177
		 'logical_or_expression', 1,
sub
#line 419 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 178
		 'logical_or_expression', 3,
sub
#line 420 "share/bi.yp"
{ $_[0]->binary_operator($_[1], $_[2], $_[3]) }
	],
	[#Rule 179
		 'conditional_expression', 1,
sub
#line 424 "share/bi.yp"
{ $_[1] }
	],
	[#Rule 180
		 'conditional_expression', 5,
sub
#line 425 "share/bi.yp"
{ $_[0]->ternary_operator($_[1], $_[2], $_[3], $_[4], $_[5]) }
	],
	[#Rule 181
		 'expression', 1,
sub
#line 429 "share/bi.yp"
{ $_[0]->expression($_[1]) }
	]
],
                                  @_);
    bless($self,$class);
}

#line 432 "share/bi.yp"


1;

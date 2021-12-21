# bigquery_view_parser.py
#
# A parser to extract table names from BigQuery view definitions.
# This is based on the `select_parser.py` sample in pyparsing:
# https://github.com/pyparsing/pyparsing/blob/master/examples/select_parser.py
#
# Michael Smedberg
#

import sys

from pyparsing import ParserElement, Suppress, Forward, CaselessKeyword
from pyparsing import MatchFirst, alphas, alphanums, Combine, Word
from pyparsing import QuotedString, CharsNotIn, Optional, Group, ZeroOrMore
from pyparsing import oneOf, delimitedList, restOfLine, cStyleComment
from pyparsing import infixNotation, opAssoc, Regex, nums

from collections import namedtuple

ColumnDescription = namedtuple('ColumnDescription', 'table name expression')
TableDescription = namedtuple('TableDescription', 'fullname cteVar')
ColumnLineage = namedtuple('ColumnLineage', 'table column')


class BigQueryViewParser:
    """Parser to extract table info from BigQuery view definitions"""
    _parser = None
    _table_identifiers = set()
    _with_aliases = set()
    _external_query_name = None

    _current_col_deps = dict()
    _final_col_deps = []
    _current_with_id = None


    def get_col_lineage(self, sql_stmt):
        return self._parse_cols(sql_stmt)

    def get_table_names(self, sql_stmt):
        table_identifiers, with_aliases = self._parse(sql_stmt)

        # Table names and alias names might differ by case, but that's not
        # relevant- aliases are not case sensitive
        lower_aliases = BigQueryViewParser.lowercase_set_of_tuples(with_aliases)
        tables = set([
            x for x in table_identifiers
            if not BigQueryViewParser.lowercase_of_tuple(x) in lower_aliases
        ])

        # Table names ARE case sensitive as described at
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#case_sensitivity
        return tables

    def _parse(self, sql_stmt):
        BigQueryViewParser._table_identifiers.clear()
        BigQueryViewParser._with_aliases.clear()
        BigQueryViewParser._get_parser().parseString(sql_stmt)

        return (BigQueryViewParser._table_identifiers, BigQueryViewParser._with_aliases)

    def _parse_cols(self, sql_stmt):
        BigQueryViewParser._current_col_deps.clear()
        BigQueryViewParser._final_col_deps.clear()
        BigQueryViewParser._get_parser().parseString(sql_stmt)
        return BigQueryViewParser._final_col_deps

    @classmethod
    def lowercase_of_tuple(cls, tuple_to_lowercase):
        return tuple(x.lower() if x else None for x in tuple_to_lowercase)

    @classmethod
    def lowercase_set_of_tuples(cls, set_of_tuples):
        return set([BigQueryViewParser.lowercase_of_tuple(x) for x in set_of_tuples])


    @classmethod
    def _get_parser(cls):
        if cls._parser is not None:
            return cls._parser

        def useTableForColLineage(t):
            # get table
            identifier_list = []
            if t.get("table", None):
                identifier_list = t.asList()
            else:
                identifier_list = t.asList()[0].split('.')
            # get rid of AS
            if len(identifier_list) > 2 and identifier_list[-2] == "AS":
                identifier_list = identifier_list[:-2]
            # get alias
            table_alias = None
            if t.asDict().get("table_alias", None):
                table_alias = t["table_alias"][0][0]
            # replace
            replace = {}
            for k,v in BigQueryViewParser._current_col_deps.items():
                if v.table.fullname == table_alias or v.table.fullname is None: # TODO is None needs to also check that column k is indeed column in table identifier_list 
                    replace[k] = (".".join(identifier_list), v.column)
            for k, v in replace.items():
                BigQueryViewParser._current_col_deps[k] = ColumnLineage(TableDescription(v[0], (None, None, v[0]) in cls._with_aliases) ,v[1])

        def replaceCTEVar(x):
            for k,v in BigQueryViewParser._current_col_deps.items():
                if k.table == x.table.fullname and k.name == x.column:
                    if not v.table.cteVar:
                        return v
                    else:
                        return replaceCTEVar(v)


        def finaliseColLineage():
            for k,v in BigQueryViewParser._current_col_deps.items():
                if k.table != "RES":
                    continue
                if not v.table.cteVar:
                    cls._final_col_deps.append( (k, v) )
                else:
                    cls._final_col_deps.append( (k, replaceCTEVar(v)) )

        def getResColLineage(t):
            if t[0].asDict().get("column", None):
                col_full_name = t[0].asDict()['column']
                col_name = col_full_name[-1]
                source_col_name = col_name
                source_col_table = col_full_name[-2] if len(col_full_name) > 1 else None

                if t[0].asDict().get("col_alias", None):
                    col_name = t[0]["col_alias"][0]

                col_expr = '???'
                if t[0].asDict().get("quoted_expr", None):
                    col_expr = t[0]["quoted_expr"].asList()
                col = ColumnDescription(cls._current_with_id or "RES", col_name, '.'.join(col_expr)) # TODO RES should be replaced with CE WITH name var within WITH
                BigQueryViewParser._current_col_deps[col] = ColumnLineage(TableDescription(source_col_table, (None, None, source_col_table) in cls._with_aliases), source_col_name)

        ParserElement.enablePackrat()

        LPAR, RPAR, COMMA, LBRACKET, RBRACKET, LT, GT = map(Suppress, "(),[]<>")
        ungrouped_select_stmt = Forward().setName("select statement")

        # keywords
        (
            ADDDATE, ALL, AND, ANY, ANY_VALUE, ARRAY, ARRAY_AGG, ARRAY_CONCAT_AGG, AS, ASC,
            ASSERT_ROWS_MODIFIED, AT, AVG, BETWEEN, BIT_AND, BIT_OR, BIT_XOR, BOOL, BY, BYTES,
            CASE, CAST, COLLATE, CONTAINS, CORR, COUNT, COUNTIF, COVAR_POP, COVAR_SAMP, CREATE,
            CROSS, CUBE, CUME_DIST, CURRENT, CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP,
            DATE, DATE_ADD, DATE_SUB, DATETIME, DEFAULT, DEFINE, DENSE_RANK, DESC, DISTINCT,
            ELSE, END, ENUM, ESCAPE, EXCEPT, EXCLUDE, EXISTS, EXTERNAL_QUERY, EXTRACT, FALSE,
            FETCH, FIRST_VALUE, FLOAT64, FOLLOWING, FOR, FROM, FULL, GENERATE_ARRAY,
            GENERATE_DATE_ARRAY, GENERATE_TIMESTAMP_ARRAY, GEOGRAPHY, GLOB, GROUP, GROUPING,
            GROUPS, HASH, HAVING, IF, IGNORE, IN, INDEXED, INFORMATION_SCHEMA, INNER, INT64,
            INTERSECT, INTERVAL, INTO, IS, ISNULL, JOIN, LAG, LAST_VALUE, LATERAL, LEAD, LEFT,
            LIKE, LIMIT, LOGICAL_AND, LOGICAL_OR, LOOKUP, MATCH, MAX, MERGE, MIN, NATURAL, NEW,
            NO, NOT, NOTNULL, NTH_VALUE, NTILE, NULL, NULLS, NUMERIC, OF, OFFSET, ON, OR, ORDER,
            ORDINAL, OUTER, OVER, PARTITION, PERCENT_RANK, PERCENTILE_CONT, PRECEDING,
            PRECENTILE_DISC, PIVOT, PROTO, QUALIFY, RANGE, RANK, RECURSIVE, REGEXP, REGEXP_EXTRACT,
            REPLACE, RESPECT, RIGHT, ROLLUP, ROW, ROW_NUMBER, ROWS, SAFE_CAST, SAFE_OFFSET,
            SAFE_ORDINAL, SELECT, SET, SOME, STDDEV, STDDEV_POP, STDDEV_SAMP, STRING_AGG, STRUCT,
            SUBDATE, SUM, SYSTEMTIME, TABLESAMPLE, THEN, TIME, TIMESTAMP, TIMESTAMP_ADD,
            TIMESTAMP_SUB, TO, TREAT, TRUE, UNBOUNDED, UNION, UNNEST, USING, VAR_POP, VAR_SAMP,
            VARIANCE, WHEN, WHERE, WINDOW, WITH, WITHIN
        ) = map(CaselessKeyword,
                """
            ADDDATE, ALL, AND, ANY, ANY_VALUE, ARRAY, ARRAY_AGG, ARRAY_CONCAT_AGG, AS, ASC,
            ASSERT_ROWS_MODIFIED, AT, AVG, BETWEEN, BIT_AND, BIT_OR, BIT_XOR, BOOL, BY, BYTES,
            CASE, CAST, COLLATE, CONTAINS, CORR, COUNT, COUNTIF, COVAR_POP, COVAR_SAMP, CREATE,
            CROSS, CUBE, CUME_DIST, CURRENT, CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP,
            DATE, DATE_ADD, DATE_SUB, DATETIME, DEFAULT, DEFINE, DENSE_RANK, DESC, DISTINCT,
            ELSE, END, ENUM, ESCAPE, EXCEPT, EXCLUDE, EXISTS, EXTERNAL_QUERY, EXTRACT, FALSE,
            FETCH, FIRST_VALUE, FLOAT64, FOLLOWING, FOR, FROM, FULL, GENERATE_ARRAY,
            GENERATE_DATE_ARRAY, GENERATE_TIMESTAMP_ARRAY, GEOGRAPHY, GLOB, GROUP, GROUPING,
            GROUPS, HASH, HAVING, IF, IGNORE, IN, INDEXED, INFORMATION_SCHEMA, INNER, INT64,
            INTERSECT, INTERVAL, INTO, IS, ISNULL, JOIN, LAG, LAST_VALUE, LATERAL, LEAD, LEFT,
            LIKE, LIMIT, LOGICAL_AND, LOGICAL_OR, LOOKUP, MATCH, MAX, MERGE, MIN, NATURAL, NEW,
            NO, NOT, NOTNULL, NTH_VALUE, NTILE, NULL, NULLS, NUMERIC, OF, OFFSET, ON, OR, ORDER,
            ORDINAL, OUTER, OVER, PARTITION, PERCENT_RANK, PERCENTILE_CONT, PRECEDING,
            PRECENTILE_DISC, PIVOT, PROTO, QUALIFY, RANGE, RANK, RECURSIVE, REGEXP, REGEXP_EXTRACT,
            REPLACE, RESPECT, RIGHT, ROLLUP, ROW, ROW_NUMBER, ROWS, SAFE_CAST, SAFE_OFFSET,
            SAFE_ORDINAL, SELECT, SET, SOME, STDDEV, STDDEV_POP, STDDEV_SAMP, STRING_AGG, STRUCT,
            SUBDATE, SUM, SYSTEMTIME, TABLESAMPLE, THEN, TIME, TIMESTAMP, TIMESTAMP_ADD,
            TIMESTAMP_SUB, TO, TREAT, TRUE, UNBOUNDED, UNION, UNNEST, USING, VAR_POP, VAR_SAMP,
            VARIANCE, WHEN, WHERE, WINDOW, WITH, WITHIN
                 """.replace(",", "").split())

        keyword = MatchFirst((
            ALL, AND, ANY, ARRAY, AS, ASC, ASSERT_ROWS_MODIFIED, AT, BETWEEN, BY, CASE, CAST,
            COLLATE, CONTAINS, CREATE, CROSS, CUBE, CURRENT, DEFAULT, DEFINE, DESC, DISTINCT,
            ELSE, END, ENUM, ESCAPE, EXCEPT, EXCLUDE, EXISTS, EXTERNAL_QUERY, EXTRACT, FALSE,
            FETCH, FOLLOWING, FOR, FROM, FULL, GROUP, GROUPING, GROUPS, HASH, HAVING, IF,
            IGNORE, IN, INNER, INTERSECT, INTERVAL, INTO, IS, JOIN, LATERAL, LEFT, LIKE, LIMIT,
            LOOKUP, MERGE, NATURAL, NEW, NO, NOT, NULL, NULLS, OF, ON, OR, ORDER, OUTER, OVER,
            PARTITION, PIVOT, PRECEDING, PROTO, RANGE, RECURSIVE, RESPECT, RIGHT, ROLLUP, ROWS,
            SELECT, SET, SOME, STRUCT, TABLESAMPLE, THEN, TO, TREAT, TRUE, UNBOUNDED, UNION,
            UNNEST, USING, WHEN, WHERE, WINDOW, WITH, WITHIN))

        # those are keywords that are fiunction names
        keyword_funcs = MatchFirst((
            TREAT, WITHIN, TABLESAMPLE, SOME, SET, RIGHT, PROTO, MERGE,
            LOOKUP, LEFT, LATERAL, FETCH, CURRENT
        ))

        identifier_word = Word(alphas + '_@#', alphanums + '@$#_')
        identifier = ~keyword + identifier_word.copy()
        collation_name = identifier.copy()
        # NOTE: Column names can't be keywords unless they are quoted
        column_name = identifier.copy() | Suppress('`') + identifier_word + Suppress('`')
        # first part of multi part column name can't be keyword, other parts can
        qualified_column_name = column_name + (
            ZeroOrMore(' ') + Suppress('.') + ZeroOrMore(' ') + identifier_word) * (0, 6)
        qualified_column_name = qualified_column_name | Suppress('`') + qualified_column_name + Suppress('`')
        # NOTE: As with column names, column aliases can be keywords, e.g. functions like `current_time`.  Other
        # keywords, e.g. `from` make parsing pretty difficult (e.g. "SELECT a from from b" is confusing.)
        # We will specifically exclude `from`, since we need to support trailing commas in the SELECT list, and
        # SQL like `SELECT a, FROM b` becomes ambiguous if we support `from`.  In that SQL, is `FROM` a column name with
        # alias `b`, or are we selecting a single column from table `b`?
        column_alias = identifier.copy()
        table_name = identifier.copy()
        table_alias = identifier.copy()
        index_name = identifier.copy()

        standard_name_part = ~keyword + Word(alphanums + "_" + "-") | keyword_funcs
        quoted_name_part = Suppress("`") + CharsNotIn("`") + Suppress("`")
        # table names can't have dots
        quoted_tablename_part = Suppress("`") + CharsNotIn("`.") + Suppress("`")

        # function_name has optional project.dataset [[project_name.]dataset_name.]function_name
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions#temporary-udf-syntax
        function_name = (Optional((quoted_name_part | standard_name_part) + Suppress('.'))
                         + Optional((quoted_name_part | standard_name_part) + Suppress('.'))
                         + (quoted_name_part | standard_name_part)
                         )
        function_name = function_name | (Suppress("`") + CharsNotIn("`") + Suppress("`"))
        parameter_name = identifier.copy()

        # expression
        expr = Forward().setName("expression")

        integer = Regex(r"[+-]?\d+")
        numeric_literal = Regex(r"[+-]?\d*\.?\d+([eE][+-]?\d+)?")
        bool_literal = TRUE | FALSE
        string_literal = (
            QuotedString("'", escChar='\\')
            | QuotedString('"', escChar='\\')
        )
        regex_literal = "r" + string_literal
        blob_literal = Regex(r"[xX]'[0-9A-Fa-f]+'")
        date_or_time_literal = (
            (DATE | TIME | DATETIME | TIMESTAMP)
            + string_literal
        )
        literal_value = (
            numeric_literal | string_literal | regex_literal | bool_literal
            | blob_literal | date_or_time_literal | NULL
            | CURRENT_TIME + Optional(LPAR + Optional(string_literal) + RPAR)
            | CURRENT_DATE + Optional(LPAR + Optional(string_literal) + RPAR)
            | CURRENT_TIMESTAMP
            + Optional(LPAR + Optional(string_literal) + RPAR)
        )
        bind_parameter = (
            Word("?", nums)
            | Combine(oneOf(": @ $") + parameter_name)
        )
        type_name = oneOf("""TEXT REAL INTEGER BLOB NULL TIMESTAMP STRING DATE
            INT64 NUMERIC FLOAT64 BOOL BYTES DATETIME GEOGRAPHY TIME ARRAY
            STRUCT""", caseless=True)
        date_part = oneOf("""MICROSECOND MILLISECOND SECOND MINUTE HOUR DAYOFWEEK
            DAY DAYOFYEAR WEEK ISOWEEK MONTH QUARTER YEAR ISOYEAR DATE TIME
            """, caseless=True)
        datetime_operators = (
            DATE_ADD | DATE_SUB | ADDDATE | SUBDATE | TIMESTAMP_ADD
            | TIMESTAMP_SUB
        )

        grouping_term = expr.copy()
        ordering_term = Group(
            expr('order_key')
            + Optional(COLLATE + collation_name('collate'))
            + Optional(ASC | DESC)('direction')
        )("ordering_term")

        function_arg = expr.copy()("function_arg")
        function_args = Optional(
            "*"
            | Optional(DISTINCT) + delimitedList(function_arg) + Optional((RESPECT | IGNORE) + NULLS) +
            Optional(ORDER + BY + delimitedList(ordering_term)) + Optional(ASC | DESC) + Optional(LIMIT + integer)
        )("function_args")
        function_call = (
            function_name("function_name")
            + LPAR + Group(function_args)("function_args_group") + RPAR
        )

        navigation_function_name = (
            FIRST_VALUE | LAST_VALUE | NTH_VALUE | LEAD | LAG
            | PERCENTILE_CONT | PRECENTILE_DISC
        )
        aggregate_function_name = (
            ANY_VALUE | ARRAY_AGG | ARRAY_CONCAT_AGG | AVG | BIT_AND | BIT_OR
            | BIT_XOR | COUNT | COUNTIF | LOGICAL_AND | LOGICAL_OR | MAX | MIN
            | STRING_AGG | SUM
        )
        statistical_aggregate_function_name = (
            CORR | COVAR_POP | COVAR_SAMP | STDDEV_POP | STDDEV_SAMP | STDDEV
            | VAR_POP | VAR_SAMP | VARIANCE
        )
        numbering_function_name = (
            RANK | DENSE_RANK | PERCENT_RANK | CUME_DIST | NTILE | ROW_NUMBER)
        analytic_function_name = (
            navigation_function_name
            | aggregate_function_name
            | statistical_aggregate_function_name
            | numbering_function_name
        )("analytic_function_name")
        partition_expression_list = delimitedList(
            grouping_term
        )("partition_expression_list")
        window_frame_boundary_start = (
            UNBOUNDED + PRECEDING
            | numeric_literal + (PRECEDING | FOLLOWING)
            | CURRENT + ROW
        )
        window_frame_boundary_end = (
            UNBOUNDED + FOLLOWING
            | numeric_literal + (PRECEDING | FOLLOWING)
            | CURRENT + ROW
        )
        window_frame_clause = (ROWS | RANGE) + (
            (
                (UNBOUNDED + PRECEDING)
                | (numeric_literal + PRECEDING)
                | (CURRENT + ROW)
            ) |
            (
                BETWEEN + window_frame_boundary_start
                + AND + window_frame_boundary_end
            ))
        window_name = identifier.copy()("window_name")
        window_specification = (
            Optional(window_name)
            + Optional(PARTITION + BY + partition_expression_list)
            + Optional(ORDER + BY + delimitedList(ordering_term))
            + Optional(window_frame_clause)("window_specification")
        )
        analytic_function = (
            analytic_function_name
            + LPAR + function_args + RPAR
            + OVER + (window_name | LPAR + Optional(window_specification) + RPAR)
        )("analytic_function")

        string_agg_term = (
            STRING_AGG
            + LPAR
            + Optional(DISTINCT)
            + expr
            + Optional(COMMA + string_literal)
            + Optional(
                ORDER + BY + expr
                + Optional(ASC | DESC)
                + Optional(LIMIT + integer)
            )
            + RPAR
        )("string_agg")
        array_literal = (
            Optional(ARRAY + Optional(LT + delimitedList(type_name) + GT))
            + LBRACKET
            + delimitedList(expr)
            + RBRACKET
        )
        interval = INTERVAL + expr + date_part
        array_generator = (
            GENERATE_ARRAY
            + LPAR
            + numeric_literal
            + COMMA
            + numeric_literal
            + COMMA
            + numeric_literal
            + RPAR
        )
        date_array_generator = (
            (GENERATE_DATE_ARRAY | GENERATE_TIMESTAMP_ARRAY)
            + LPAR
            + expr("start_date")
            + COMMA
            + expr("end_date")
            + Optional(COMMA + interval)
            + RPAR
        )

        explicit_struct = (
            STRUCT
            + Optional(LT + delimitedList(type_name) + GT)
            + LPAR
            + Optional(delimitedList(expr + Optional(AS + identifier)))
            + RPAR
        )

        case_when = WHEN + expr.copy()("when")
        case_then = THEN + expr.copy()("then")
        case_clauses = Group(ZeroOrMore((case_when + case_then)))
        case_else = ELSE + expr.copy()("else")
        case_stmt = (
            CASE
            + Optional(expr.copy())
            + case_clauses("case_clauses")
            + Optional(case_else) + END
        )("case")
        if_expr = IF + LPAR + expr + COMMA + expr + COMMA + expr + RPAR
        expr_term = (
            (analytic_function)("analytic_function")
            | (CAST + LPAR + expr + AS + type_name + RPAR)("cast")
            | (SAFE_CAST + LPAR + expr + AS + type_name + RPAR)("safe_cast")
            | (
                Optional(EXISTS)
                + LPAR + ungrouped_select_stmt + RPAR
            )("subselect")
            | (literal_value)("literal")
            | (bind_parameter)("bind_parameter")
            | (EXTRACT + LPAR + expr + FROM + expr + RPAR)("extract")
            | case_stmt
            | (
                datetime_operators
                + LPAR + expr + COMMA + interval + RPAR
            )("date_operation")
            | string_agg_term("string_agg_term")
            | array_literal("array_literal")
            | array_generator("array_generator")
            | date_array_generator("date_array_generator")
            | explicit_struct("explicit_struct")
            | function_call("function_call")
            | qualified_column_name("column")
            | if_expr
        ) + Optional(
            LBRACKET
            + (OFFSET | ORDINAL | SAFE_OFFSET | SAFE_ORDINAL)
            + LPAR + expr + RPAR
            + RBRACKET
        )("offset_ordinal")

        struct_term = (LPAR + delimitedList(expr_term) + RPAR)

        UNARY, BINARY, TERNARY = 1, 2, 3
        expr << infixNotation((expr_term | struct_term), [
            (oneOf('- + ~') | NOT, UNARY, opAssoc.RIGHT),
            (ISNULL | NOTNULL | NOT + NULL, UNARY, opAssoc.LEFT),
            ('||', BINARY, opAssoc.LEFT),
            (oneOf('* / %'), BINARY, opAssoc.LEFT),
            (oneOf('+ -'), BINARY, opAssoc.LEFT),
            (oneOf('<< >> & |'), BINARY, opAssoc.LEFT),
            (oneOf("= > < >= <= <> != !< !>"), BINARY, opAssoc.LEFT),
            (
                IS + Optional(NOT)
                | Optional(NOT) + IN + Optional(UNNEST)
                | Optional(NOT) + LIKE
                | GLOB
                | MATCH
                | REGEXP, BINARY, opAssoc.LEFT
            ),
            ((BETWEEN, AND), TERNARY, opAssoc.LEFT),
            (
                Optional(NOT) + IN
                + LPAR
                + Group(ungrouped_select_stmt | delimitedList(expr))
                + RPAR,
                UNARY,
                opAssoc.LEFT
            ),
            (AND, BINARY, opAssoc.LEFT),
            (OR, BINARY, opAssoc.LEFT),
        ])
        quoted_expr = (
            expr
            ^ Suppress('"') + expr + Suppress('"')
            ^ Suppress("'") + expr + Suppress("'")
        )("quoted_expr")

        compound_operator = (
            UNION + Optional(ALL | DISTINCT)
            | INTERSECT + DISTINCT
            | EXCEPT + DISTINCT
            | INTERSECT
            | EXCEPT
        )("compound_operator")

        join_constraint = Group(
            Optional(
                ON + expr
                | USING + LPAR
                + Group(delimitedList(qualified_column_name))
                + RPAR
            ))("join_constraint")

        join_op = (
            COMMA
            | Group(
                Optional(NATURAL)
                + Optional(
                    INNER
                    | CROSS
                    | LEFT + OUTER
                    | LEFT
                    | RIGHT + OUTER
                    | RIGHT
                    | FULL + OUTER
                    | OUTER
                    | FULL
                ) + JOIN
            )
        )("join_op")

        join_source = Forward()

        # We support a few kinds of table identifiers.
        #
        # First, dot delimited info like project.dataset.table, where
        # each component follows the rules described in the BigQuery
        # docs, namely:
        #  Contain letters (upper or lower case), numbers, and underscores
        #
        # Second, a dot delimited quoted string.  Since it's quoted, we'll be
        # liberal w.r.t. what characters we allow.  E.g.:
        #  `project.dataset.name-with-dashes`
        #
        # Third, a series of quoted strings, delimited by dots, e.g.:
        #  `project`.`dataset`.`name-with-dashes`
        #
        # We also support combinations, like:
        #  project.dataset.`name-with-dashes`
        #  `project`.`dataset.name-with-dashes`
        #
        # In some cases, the identifier might include more than 3 dots.
        # Metadata view names include dots, e.g.
        # project.dataset.INFORMATION_SCHEMA.TABLES.
        # In this case, the trailing information is a "table" we're selecting
        # from.

        def record_quoted_table_identifier(t):
            identifier_list = t.asList()[0].split('.')
            # If the next to last item is "INFORMATION_SCHEMA", then combine
            # it with the last item; they're essentially the view name.
            if (len(identifier_list) > 1) and (identifier_list[-2].upper() == "INFORMATION_SCHEMA"):
                identifier_list[-2] = identifier_list[-2] + "." + identifier_list[-1]
                del identifier_list[-1]

            first = ".".join(identifier_list[0:-2]) or None
            second = identifier_list[-2]
            third = identifier_list[-1]
            identifier_list = [first, second, third]
            padded_list = [None] * (3 - len(identifier_list)) + identifier_list
            cls._table_identifiers.add(tuple(padded_list))

        def record_unquoted_table_identifier(t):
            identifier_list = t.asList()
            if cls._external_query_name is not None:
                if len(identifier_list) != 1:
                    raise Exception(
                        (f"_external_query_name is {cls._external_query_name} but identifier_list is not only " +
                         f"table name: {identifier_list}"))
                else:
                    identifier_list.insert(0, cls._external_query_name)
            padded_list = [None] * (3 - len(identifier_list)) + identifier_list
            # If padded list has more than 3 elements, combine the "trailing"
            # elements into a single identifier
            if len(padded_list) > 3:
                padded_list = [padded_list[0], padded_list[1], ".".join(padded_list[2:])]
            cls._table_identifiers.add(tuple(padded_list))

        quoted_table_parts_identifier = (
            Optional((quoted_name_part.copy()("project") | standard_name_part.copy()("project")) + Suppress('.'))
            + Optional((quoted_name_part.copy()("dataset") | standard_name_part.copy()("dataset")) + Suppress('.'))
            + Optional(INFORMATION_SCHEMA + Suppress('.'))
            + (quoted_tablename_part.copy()("table") | standard_name_part.copy()("table"))
        ).setParseAction(lambda t: record_unquoted_table_identifier(t))

        quotable_table_parts_identifier = (
            Suppress("`") + CharsNotIn("`") + Suppress("`")
        ).setParseAction(lambda t: record_quoted_table_identifier(t))

        table_identifier = (
            quoted_table_parts_identifier |
            quotable_table_parts_identifier
        )

        def unset_external_query_name(tokens):
            cls._external_query_name = None

        def set_external_query_name(tokens):
            if cls._external_query_name is not None:
                raise Exception(
                    (f"external_query_name value is {cls._external_query_name} and trying to set it to {tokens[0]}." +
                     " Nested external queries?"))
            cls._external_query_name = tokens[0]

        external_query = (EXTERNAL_QUERY + LPAR + QuotedString('"').setParseAction(set_external_query_name) + ","
                          + Suppress('"') + ungrouped_select_stmt + Suppress('"')
                          + RPAR).setParseAction(unset_external_query_name)

        single_source = (
            (
                (
                    table_identifier
                    + Optional(Optional(AS) + table_alias("table_alias*"))
                    + Optional(FOR + SYSTEMTIME + AS + OF + string_literal)
                    + Optional(
                        INDEXED + BY + index_name("name")
                        | NOT + INDEXED
                    )
                )("index")
                | (
                    LPAR
                    + ungrouped_select_stmt
                    + RPAR
                )
                | (LPAR + join_source + RPAR)
                | (UNNEST + LPAR + expr + RPAR)
                | external_query
            )
            + Optional(Optional(AS) + table_alias)
        ).setParseAction(useTableForColLineage)

        join_source << single_source + ZeroOrMore(join_op + single_source + join_constraint)

        over_partition = (
            PARTITION + BY
            + delimitedList(partition_expression_list)
        )("over_partition")
        over_order = (ORDER + BY + delimitedList(ordering_term))
        over_unsigned_value_specification = expr
        over_window_frame_preceding = (
            UNBOUNDED + PRECEDING
            | over_unsigned_value_specification + PRECEDING
            | CURRENT + ROW
        )
        over_window_frame_following = (
            UNBOUNDED + FOLLOWING
            | over_unsigned_value_specification + FOLLOWING
            | CURRENT + ROW
        )
        over_window_frame_bound = (
            over_window_frame_preceding
            | over_window_frame_following
        )
        over_window_frame_between = (
            BETWEEN + over_window_frame_bound + AND + over_window_frame_bound
        )
        over_window_frame_extent = (
            over_window_frame_preceding
            | over_window_frame_between
        )
        over_row_or_range = ((ROWS | RANGE) + over_window_frame_extent)
        over = (
            OVER
            + LPAR
            + Optional(over_partition)
            + Optional(over_order)
            + Optional(over_row_or_range)
            + RPAR
        )("over")

        replace_col_expr = expr + Optional(AS) + column_name

        result_column = (
            Optional(table_name + ".")
            + "*"
            + Optional(EXCEPT + LPAR + delimitedList(column_name) + RPAR)
            + Optional(REPLACE + LPAR + delimitedList(replace_col_expr) + RPAR)
            | Group(
                # Disallow selecting a column called `FROM`, since it makes SQL like "SELECT a, b, FROM c" ambiguous.
                # NOTE: This may be true for `GROUP BY` and other keywords as well.
                (~FROM + quoted_expr)
                + Optional(over)
                + Optional(Optional(AS) + column_alias("col_alias"))
            )
        ).setParseAction(getResColLineage)

        window_select_clause_specification = identifier + AS + LPAR + window_specification + RPAR
        window_select_clause = WINDOW + delimitedList(window_select_clause_specification)

        with_stmt = Forward().setName("with statement")
        ungrouped_select_no_with = (
            SELECT
            + Optional(AS + STRUCT)
            + Optional(DISTINCT | ALL)
            + Group(delimitedList(result_column))("columns")
            + Optional(COMMA)  # To support trailing commas, like "SELECT a, b, FROM x"
            + Optional(FROM + join_source("from*"))
            + Optional(WHERE + expr)
            + Optional(QUALIFY + expr)
            + Optional(
                GROUP + BY
                + Group(delimitedList(grouping_term))("group_by_terms")
            )
            + Optional(HAVING + expr("having_expr"))
            + Optional(window_select_clause)
            + Optional(
                ORDER + BY
                + Group(delimitedList(ordering_term))("order_by_terms")
            )
        )
        select_no_with = ungrouped_select_no_with | (LPAR + ungrouped_select_no_with + RPAR)
        select_core = (
            Optional(with_stmt)
            + select_no_with
        )
        grouped_select_core = select_core | (LPAR + select_core + RPAR)

        agg_function_call = aggregate_function_name + LPAR + function_args + RPAR + Optional(AS + column_alias)
        pivot_clause = (
            PIVOT + LPAR + delimitedList(agg_function_call) + FOR
            + column_name + IN + LPAR + delimitedList(string_literal) + RPAR
            + RPAR + Optional(AS + column_alias)
        )

        ungrouped_select_stmt << (
            grouped_select_core
            + ZeroOrMore(compound_operator + grouped_select_core)
            + Optional(
                LIMIT
                + (
                    Group(expr + OFFSET + expr)
                    | Group(expr + COMMA + expr)
                    | expr
                )("limit")
            )
            + Optional(pivot_clause)
        )("select")
        select_stmt = ungrouped_select_stmt | (LPAR + ungrouped_select_stmt + RPAR)
        select_stmt.setParseAction(finaliseColLineage)
        # define comment format, and ignore them
        sql_comment = (oneOf("-- #") + restOfLine | cStyleComment)
        select_stmt.ignore(sql_comment)

        def record_with_alias(t):
            identifier_list = t.asList()
            padded_list = [None] * (3 - len(identifier_list)) + identifier_list
            cls._current_with_id = padded_list[2]
            cls._with_aliases.add(tuple(padded_list))

        def clear_with_scope():
            cls._current_with_id = None

        with_clause = Group(
            identifier.setParseAction(lambda t: record_with_alias(t))
            + AS
            + LPAR + select_stmt + RPAR
        ).setParseAction(clear_with_scope)
        with_stmt << (WITH + delimitedList(with_clause))
        with_stmt.ignore(sql_comment)

        cls._parser = select_stmt
        return cls._parser

    TEST_CASES_COLS = [
        [
            """
            SELECT a.c1,a.c2 FROM table AS a
            """,
            [
                (ColumnDescription('RES', 'c1', 'a.c1'), ColumnLineage(TableDescription('table', False), 'c1')),
                (ColumnDescription('RES', 'c2', 'a.c2'), ColumnLineage(TableDescription('table', False), 'c2')),
            ]
        ],
        [
            """
            SELECT d1, d2 FROM table
            """,
            [
                (ColumnDescription('RES', 'd1', 'd1'), ColumnLineage(TableDescription('table', False), 'd1')),
                (ColumnDescription('RES', 'd2', 'd2'), ColumnLineage(TableDescription('table', False), 'd2')),
            ]
        ],
        [
            """
            SELECT a.e1 AS ee1, e2 FROM table AS a
            """,
            [
                (ColumnDescription('RES', 'ee1', 'a.e1'), ColumnLineage(TableDescription('table', False), 'e1')),
                (ColumnDescription('RES', 'e2', 'e2'), ColumnLineage(TableDescription('table', False), 'e2')),
            ]
        ],
        [
            """
            SELECT a.f1 AS ff1, b.f2 AS ff2 FROM table1 AS a JOIN table2 AS b ON a.c1 = b.c1
            """,
            [
                (ColumnDescription('RES', 'ff1', 'a.f1'), ColumnLineage(TableDescription('table1', False), 'f1')),
                (ColumnDescription('RES', 'ff2', 'b.f2'), ColumnLineage(TableDescription('table2', False), 'f2')),
            ]
        ],
        [
            """
            SELECT a.g1 AS gg1, b.g2 AS gg2 FROM ds1.table1 AS a JOIN `ds2.table2` AS b ON a.c1 = b.c1
            """,
            [
                (ColumnDescription('RES', 'gg1', 'a.g1'), ColumnLineage(TableDescription('ds1.table1', False), 'g1')),
                (ColumnDescription('RES', 'gg2', 'b.g2'), ColumnLineage(TableDescription('ds2.table2', False), 'g2')),
            ]
        ],
        [   # TODO this is not always correct, unless we are sure that g1 is column in table1 (not table2)
            """
            SELECT g1 AS gg1, b.g2 AS gg2 FROM ds1.table1 AS a JOIN `ds2.table2` AS b ON a.c1 = b.c1
            """,
            [
                (ColumnDescription('RES', 'gg1', 'g1'), ColumnLineage(TableDescription('ds1.table1', False), 'g1')),
                (ColumnDescription('RES', 'gg2', 'b.g2'), ColumnLineage(TableDescription('ds2.table2', False), 'g2')),
            ]
        ],
        [
            """
            WITH x AS (SELECT a.c FROM table AS a)
            SELECT x.c FROM x
            """,
            [
                (ColumnDescription('RES', 'c', 'x.c'), ColumnLineage(TableDescription('table', False), 'c')),
            ]
        ],
        [
            """
            WITH x AS (SELECT a.c FROM table AS a)
            SELECT t.c FROM x AS t
            """,
            [
                (ColumnDescription('RES', 'c', 't.c'), ColumnLineage(TableDescription('table', False), 'c')),
            ]
        ],
        [
            """
            WITH x AS (SELECT a.c FROM table AS a),
            y AS (SELECT x.c FROM x)
            SELECT t.c FROM y AS t
            """,
            [
                (ColumnDescription('RES', 'c', 't.c'), ColumnLineage(TableDescription('table', False), 'c')),
            ]
        ],
        [
            """
            WITH x AS (SELECT a.c, a.d FROM table AS a)
            SELECT t.c, t.d FROM x AS t
            """,
            [
                (ColumnDescription('RES', 'c', 't.c'), ColumnLineage(TableDescription('table', False), 'c')),
                (ColumnDescription('RES', 'd', 't.d'), ColumnLineage(TableDescription('table', False), 'd')),
            ]
        ],
    ]
    def test_cols(self):
        for test_index, test_case in enumerate(BigQueryViewParser.TEST_CASES_COLS):
            sql_stmt, expected_col_lineage = test_case

            cols_lineage = self.get_col_lineage(sql_stmt)

            if expected_col_lineage != cols_lineage:
                raise Exception(f"Test {test_index} failed- expected {expected_col_lineage} but got {cols_lineage}")


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    BigQueryViewParser().test_cols()

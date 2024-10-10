import random
from sympy import sympify
from typing import Dict


class Equation:
    operators = ['+', '-', '*', '/']
    operator_names = ['add', 'sub', 'mul', 'div']
    op_dict = dict(zip(operators, operator_names))
    op_reverse_dict = dict(zip(operator_names, operators))

    def __init__(self, num_operand: int, num_operator: int):
        assert num_operand == num_operator + 1, "Number of operators must be one less than number of operands"
        self.num_operand = num_operand
        self.num_operator = num_operator

    @staticmethod
    def get_lhs_rhs(equation: str) -> (str, str):
        lhs, rhs = equation.split("=")
        return lhs, rhs

    @staticmethod
    def get_equation_answer(equation: str, operator: str):
        if operator == 'x':
            mo_equation = equation.replace("x", "*")
        elif operator == 'minus':
            mo_equation = equation.replace("minus", "-")
        else:
            mo_equation = equation
        # mo_equation = equation.replace("\\minus", "-") if "\\minus" in equation else equation
        answer = sympify(mo_equation).evalf(3) if operator == "/" else sympify(mo_equation)  # 3 decimal places
        return answer


class MultiDigitEquation(Equation):

    def __init__(self, num_operator, num_digit):
        assert num_digit > 0, "Number of digits must be greater than 0"
        num_operand = num_operator + 1
        super().__init__(num_operand, num_operator)

    def generate_equation(self, operator: str, input_range: Dict, int_only: bool = False) -> str:
        raise NotImplementedError


class MultiOpEquation(Equation):

    def __init__(self, num_operator):
        assert num_operator >= 1, "Number of operators must be greater than equal to 1"
        num_operand = num_operator + 1
        super().__init__(num_operand, num_operator)

    def generate_equation(self, operator, input_range: Dict, int_only=False, ans_wrap=None) -> str:
        assert operator in self.operators, f"Operator must be one of {self.operators}"
        operands = [random.randint(input_range['start'], input_range['end']) for _ in range(self.num_operand)]
        operators = [operator for _ in range(self.num_operator)]

        lhs = " ".join([str(operands[i]) + " " + operators[i] for i in range(self.num_operator)]) + " " + str(operands[-1])
        rhs = self.get_equation_answer(lhs, operator)
        ans = str(int(rhs) if int_only else f"{rhs}")
        if ans_wrap:
            eq = lhs + " = " + ans_wrap.replace("num", ans)
        else:
            eq = lhs + "=" + ans
        return eq


if __name__ == "__main__":
    ip_range = {'start': 1, 'end': 4}
    multi_op_eq = MultiOpEquation(num_operator=1)
    eq = multi_op_eq.generate_equation('+', ip_range, int_only=True)
    print(eq)

    multi_op_eq = MultiOpEquation(num_operator=2)
    eq = multi_op_eq.generate_equation('-', ip_range, int_only=True)
    print(eq)

    multi_op_eq = MultiOpEquation(num_operator=1)
    eq = multi_op_eq.generate_equation('*', ip_range, int_only=True)
    print(eq)

    multi_op_eq = MultiOpEquation(num_operator=1)
    eq = multi_op_eq.generate_equation('/', ip_range, int_only=False)
    print(eq)

import math
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP
from PIL import Image as PILImage

try:
    from .models import (
        AddInput, AddOutput, SubtractInput, SubtractOutput,
        MultiplyInput, MultiplyOutput, DivideInput, DivideOutput,
        PowerInput, PowerOutput, CbrtInput, CbrtOutput,
        FactorialInput, FactorialOutput, RemainderInput, RemainderOutput,
        SinInput, SinOutput, CosInput, CosOutput, TanInput, TanOutput,
        MineInput, MineOutput, CreateThumbnailInput, ImageOutput,
        StringsToIntsInput, StringsToIntsOutput, ExpSumInput, ExpSumOutput,
        FibonacciInput, FibonacciOutput, CurrentTimeInput, CurrentTimeOutput,
    )
    from .server_utils import run_mcp_server
except ImportError:
    from models import (
        AddInput, AddOutput, SubtractInput, SubtractOutput,
        MultiplyInput, MultiplyOutput, DivideInput, DivideOutput,
        PowerInput, PowerOutput, CbrtInput, CbrtOutput,
        FactorialInput, FactorialOutput, RemainderInput, RemainderOutput,
        SinInput, SinOutput, CosInput, CosOutput, TanInput, TanOutput,
        MineInput, MineOutput, CreateThumbnailInput, ImageOutput,
        StringsToIntsInput, StringsToIntsOutput, ExpSumInput, ExpSumOutput,
        FibonacciInput, FibonacciOutput, CurrentTimeInput, CurrentTimeOutput,
    )
    from server_utils import run_mcp_server

mcp = FastMCP("math")


@mcp.tool()
def add(input: AddInput) -> AddOutput:
    """Add two numbers."""
    return AddOutput(result=input.a + input.b)

@mcp.tool()
def subtract(input: SubtractInput) -> SubtractOutput:
    """Subtract one number from another."""
    return SubtractOutput(result=input.a - input.b)

@mcp.tool()
def multiply(input: MultiplyInput) -> MultiplyOutput:
    """Multiply two numbers."""
    return MultiplyOutput(result=input.a * input.b)

@mcp.tool()
def divide(input: DivideInput) -> DivideOutput:
    """Divide one number by another."""
    return DivideOutput(result=input.a / input.b)

@mcp.tool()
def power(input: PowerInput) -> PowerOutput:
    """Raise a to the power of b."""
    return PowerOutput(result=input.a ** input.b)

@mcp.tool()
def cbrt(input: CbrtInput) -> CbrtOutput:
    """Compute cube root of a number."""
    return CbrtOutput(result=input.a ** (1 / 3))

@mcp.tool()
def factorial(input: FactorialInput) -> FactorialOutput:
    """Compute factorial of a number."""
    return FactorialOutput(result=math.factorial(input.a))

@mcp.tool()
def remainder(input: RemainderInput) -> RemainderOutput:
    """Compute remainder of a divided by b."""
    return RemainderOutput(result=input.a % input.b)

@mcp.tool()
def sin(input: SinInput) -> SinOutput:
    """Compute sine of angle in radians."""
    return SinOutput(result=math.sin(input.a))

@mcp.tool()
def cos(input: CosInput) -> CosOutput:
    """Compute cosine of angle in radians."""
    return CosOutput(result=math.cos(input.a))

@mcp.tool()
def tan(input: TanInput) -> TanOutput:
    """Compute tangent of angle in radians."""
    return TanOutput(result=math.tan(input.a))

@mcp.tool()
def mine(input: MineInput) -> MineOutput:
    """Special mining operation: a - 2b."""
    return MineOutput(result=input.a - input.b - input.b)

@mcp.tool()
def current_time(input: CurrentTimeInput) -> CurrentTimeOutput:
    """Return exact current local times for one or more IANA timezones."""
    lines = []
    for timezone_name in input.timezones:
        try:
            current = datetime.now(ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError:
            lines.append(f"{timezone_name}: invalid or unavailable IANA timezone")
            continue
        lines.append(
            f"{timezone_name}: {current.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"(UTC{current.strftime('%z')[:3]}:{current.strftime('%z')[3:]})"
        )
    return CurrentTimeOutput(result="\n".join(lines))


@mcp.tool()
def create_thumbnail(input: CreateThumbnailInput) -> ImageOutput:
    """Create a 100x100 thumbnail from an image file."""
    img = PILImage.open(input.image_path)
    img.thumbnail((100, 100))
    return ImageOutput(data=img.tobytes(), format="png")

@mcp.tool()
def strings_to_chars_to_int(input: StringsToIntsInput) -> StringsToIntsOutput:
    """Convert string characters to their ASCII values."""
    return StringsToIntsOutput(ascii_values=[ord(c) for c in input.string])

@mcp.tool()
def int_list_to_exponential_sum(input: ExpSumInput) -> ExpSumOutput:
    """Sum of exponentials of a list of integers."""
    return ExpSumOutput(result=sum(math.exp(i) for i in input.numbers))

@mcp.tool()
def fibonacci_numbers(input: FibonacciInput) -> FibonacciOutput:
    """Generate first n Fibonacci numbers."""
    if input.n <= 0:
        return FibonacciOutput(result=[])
    seq = [0, 1]
    for _ in range(2, input.n):
        seq.append(seq[-1] + seq[-2])
    return FibonacciOutput(result=seq[:input.n])


if __name__ == "__main__":
    run_mcp_server(mcp)

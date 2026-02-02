"""
Command-line interface for Immersion Cooling Validator.

Usage:
    icv validate [--samples N] [--grade GRADE] [--output FILE]
    icv compliance --tj TEMP --bdv VOLTAGE --resistivity RES --life YEARS
    icv server [--port PORT]
"""

import argparse
import json
import sys


def cmd_validate(args):
    """Run Monte Carlo validation."""
    from icv import validate_fluid, GroupIIIOil
    
    fluid = GroupIIIOil(viscosity_grade=args.grade)
    result = validate_fluid(
        fluid=fluid,
        n_samples=args.samples,
        include_economics=True,
    )
    
    if args.output:
        result.save(args.output)
        print(f"Results saved to {args.output}")
    else:
        print(result.summary())
    
    return 0 if result.compliance.critical_pass else 1


def cmd_compliance(args):
    """Check OCP compliance."""
    from icv import check_ocp_compliance
    
    result = check_ocp_compliance(
        junction_temp_C=args.tj,
        breakdown_voltage_kV=args.bdv,
        volume_resistivity_ohm_cm=args.resistivity,
        p5_life_years=args.life,
    )
    
    print(f"Critical Pass: {'✅ PASS' if result.critical_pass else '❌ FAIL'}")
    for r in result.results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.name}: {r.value:.2g} {r.unit} (limit: {r.limit})")
    
    return 0 if result.critical_pass else 1


def cmd_server(args):
    """Start web server."""
    try:
        import uvicorn
        uvicorn.run(
            "icv.web.backend.main:app",
            host="0.0.0.0",
            port=args.port,
            reload=args.reload,
        )
    except ImportError:
        print("Web dependencies not installed. Run: pip install icv[web]")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="icv",
        description="Immersion Cooling Validator - Monte Carlo validation for cooling fluids",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # validate
    p_validate = subparsers.add_parser("validate", help="Run MC validation")
    p_validate.add_argument("-n", "--samples", type=int, default=10000, help="MC samples")
    p_validate.add_argument("-g", "--grade", default="4cSt", choices=["4cSt", "6cSt", "8cSt"])
    p_validate.add_argument("-o", "--output", help="Output JSON file")
    
    # compliance
    p_compliance = subparsers.add_parser("compliance", help="Check OCP compliance")
    p_compliance.add_argument("--tj", type=float, required=True, help="Junction temp (°C)")
    p_compliance.add_argument("--bdv", type=float, required=True, help="Breakdown voltage (kV)")
    p_compliance.add_argument("--resistivity", type=float, required=True, help="Resistivity (Ω·cm)")
    p_compliance.add_argument("--life", type=float, required=True, help="P5 life (years)")
    
    # server
    p_server = subparsers.add_parser("server", help="Start web server")
    p_server.add_argument("-p", "--port", type=int, default=8000)
    p_server.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "compliance":
        sys.exit(cmd_compliance(args))
    elif args.command == "server":
        sys.exit(cmd_server(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()

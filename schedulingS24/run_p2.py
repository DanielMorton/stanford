import util, submission, sys

if len(sys.argv) < 2:
    print(("Usage: %s <profile file (e.g., profile3d.txt)>" % sys.argv[0]))
    sys.exit(1)

profilePath = sys.argv[1]
bulletin = util.CourseBulletin('courses.json')
profile = util.Profile(bulletin, profilePath)
profile.print_info()
cspConstructor = submission.SchedulingCSPConstructor(bulletin, profile)
csp = cspConstructor.get_basic_csp()
cspConstructor.add_all_additional_constraints(csp)

alg = submission.BacktrackingSearch()
alg.solve(csp, mcv = True, ac3 = True)
if alg.optimalAssignment:
    print((alg.optimalWeight))
    for key, value in list(alg.optimalAssignment.items()):
        print((key, '=', value))

for assignment in alg.allOptimalAssignments:
    solution = util.extract_course_scheduling_solution(profile, assignment)
    if solution:
        # displays one of the best assignments
        util.print_course_scheduling_solution(solution)
        break
    elif alg.numOptimalAssignments == 1:
        print("The best schedule is to take 0 units every quarter.")

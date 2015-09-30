#include <array>
#include "CenterFindEngine.h"

int main(int argc, char ** argv) {
	if (argc != 13)
		return EXIT_FAILURE;

	std::array<std::string, 12> args;
	for (int i = 0; i < 12; i++)
		args[i] = std::string(argv[i]);

	CenterFindEngine::Parameters params(args);
	CenterFindEngine C(params);

	CenterFindEngine::PMetricsVec results = C.Execute();
}
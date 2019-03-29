//Tensorflow Versions
def versions = [
	"1.7.0",
	"1.8.0",
	"1.9.0",
	"1.10.0",
	"1.11.0",
	"1.12.0",
	"1.13.0"
]

def repository = 'hydro-serving-tensorflow'
def tensorflowImages = versions.collect {"hydrosphere/serving-runtime-tensorflow-${it}"}

def buildFunction={
    def curVersion = getVersion()
    versions.each {
        sh "make VERSION=${curVersion} tf-${it}"
    }
}

def collectTestResults = {
    junit testResults: '**/target/test-reports/io.hydrosphere*.xml', allowEmptyResults: true
}

pipelineCommon(
        repository,
        false, //needSonarQualityGate,
        tensorflowImages,
        collectTestResults,
        buildFunction,
        buildFunction,
        buildFunction
)

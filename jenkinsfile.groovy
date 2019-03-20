//Tensorflow Versions
def versions = [
        "1.1.0",
        "1.2.0",
        "1.3.0",
        "1.4.0",
	"1.5.0",
	"1.6.0",
	"1.7.0",
	"1.8.0",
	"1.9.0",
	"1.10.0",
	"1.11.0"
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

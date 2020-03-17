/*
 * Copyright (c) 2020 Kay Gawlik <kaydev@amarunet.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <string>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include "boost/python/numpy.hpp"

namespace bp = boost::python;
namespace bn = boost::python::numpy;

#include <opencv2/opencv.hpp>

#include <octdata/octfileread.h>
#include <octdata/filereadoptions.h>
#include <octdata/datastruct/oct.h>

#include <octdata/datastruct/sloimage.h>
#include <octdata/datastruct/bscan.h>


namespace
{
	template<typename S>
	std::string getSubStructureName()
	{
		std::string name = boost::typeindex::type_id<typename S::SubstructureType>().pretty_name();
		std::size_t namePos = name.rfind(':');
		if(namePos > 0)
			++namePos;
		return name.substr(namePos, name.size() - namePos);
	}


	template<typename T>
	bn::ndarray wrapOpenCvMat(const cv::Mat& mat)
	{
		Py_intptr_t shape[2];
		shape[0] = mat.rows;
		shape[1] = mat.cols;
		bn::ndarray result = bn::zeros(2, shape, bn::dtype::get_builtin<char>());

		T* destPtr = reinterpret_cast<T*>(result.get_data());
		for(int row = 0; row < mat.rows; ++row)
		{
			const T* const sourcePtr = reinterpret_cast<T*>(mat.data + mat.cols*row);
			std::copy(sourcePtr, sourcePtr+mat.cols, destPtr);
			destPtr += mat.cols;
		}
		return result;
	}

	class ParameterToOptions
	{
		ParameterToOptions* parent = nullptr;
		std::string nameInParent;

		bp::dict valueDict;
	public:
		~ParameterToOptions()
		{
			if(parent)
				parent->valueDict[nameInParent] = getValueDict();
		}

		template<typename T>
		void operator()(const std::string& name, T& value)
		{
			valueDict[name] = value;
		}


		template<typename T>
		void operator()(const std::string& name, const std::vector<T>& value)
		{
			bp::list list;
			for(const T& val : value)
				list.append(val);
			valueDict[name] = list;
		}


		ParameterToOptions subSet(const std::string& name)
		{
			ParameterToOptions pto;

			pto.parent = this;
			pto.nameInParent = name;

			return pto;
		}

		bp::dict getValueDict()
		{
			return std::move(valueDict);
		}

	};

	template<typename T>
	inline T getConfigFromStruct(const bp::dict& dict, const char* name, const T defaultValue)
	{
		if(dict.has_key(name))
			return bp::extract<T>(dict[name]);
		return defaultValue;
	}


	class ParameterFromOptions
	{
		const bp::dict* dict;

		ParameterFromOptions(const bp::dict* dict) : dict(dict) {}
	public:
		ParameterFromOptions(const bp::dict& dict) : dict(&dict) {}

		template<typename T>
		void operator()(const char* name, T& value)
		{
			if(dict)
				value = getConfigFromStruct(*dict, name, value);
		}

		ParameterFromOptions subSet(const std::string& name)
		{
			if(dict && dict->has_key(name))
				return ParameterFromOptions(bp::extract<bp::dict>((*dict)[name]));
			return ParameterFromOptions(nullptr);
		}
	};

	// general export methods
	bp::dict convertSlo(const OctData::SloImage& slo)
	{
		bp::dict dict;
		ParameterToOptions pto;
		slo.getSetParameter(pto);
		dict["data"] = pto.getValueDict();
		dict["image"] = wrapOpenCvMat<uint8_t>(slo.getImage());

		return dict;
	}

	bp::dict convertSegmentation(const OctData::Segmentationlines& seglines)
	{
		bp::dict dict;
		for(OctData::Segmentationlines::SegmentlineType type : OctData::Segmentationlines::getSegmentlineTypes())
		{
			const OctData::Segmentationlines::Segmentline& seg = seglines.getSegmentLine(type);
			if(!seg.empty())
			{
				bp::list segLine;
				for(double value : seg)
					segLine.append(value);
				dict[OctData::Segmentationlines::getSegmentlineName(type)] = segLine;
			}
		}

		return dict;
	}

	bp::dict convertBScan(const OctData::BScan* bscan)
	{
		if(!bscan)
			return bp::dict();

		bp::dict dict;

		ParameterToOptions pto;
		bscan->getSetParameter(pto);
		dict["data"] = pto.getValueDict();


		if(!bscan->getImage().empty())
			dict["image"] = wrapOpenCvMat<uint8_t>(bscan->getImage());
		if(!bscan->getAngioImage().empty())
			dict["imageAngio"] = wrapOpenCvMat<uint8_t>(bscan->getAngioImage());

		dict["data"] = convertSegmentation(bscan->getSegmentLines());
		return dict;
	}

	template<typename S>
	bp::dict convertStructure(const S& structure)
	{
		static const std::string structureName = getSubStructureName<S>();

		bp::dict dict;

		ParameterToOptions pto;
		structure.getSetParameter(pto);

		dict["data"] = pto.getValueDict();

		for(typename S::SubstructurePair const& subStructPair : structure)
		{
			std::string subStructName = structureName + '_' + boost::lexical_cast<std::string>(subStructPair.first);
			dict[subStructName] = convertStructure(*subStructPair.second);
		}
		return dict;
	}

	template<>
	bp::dict convertStructure<OctData::Series>(const OctData::Series& series)
	{
		bp::dict dict;

		ParameterToOptions pto;
		series.getSetParameter(pto);

		dict["data"] = pto.getValueDict();
		dict["slo"]  = convertSlo(series.getSloImage());

		bp::list bscanList;
		for(const OctData::BScan* bscan : series.getBScans())
			bscanList.append(convertBScan(bscan));

		dict["bscans"] = std::move(bscanList);
		return dict;
	}

}


bp::dict readFile(std::string filename, const OctData::FileReadOptions& options)
{
	if(filename.empty())
		return bp::dict();

	OctData::OCT oct = OctData::OctFileRead::openFile(filename, options);

	return convertStructure(oct);
}


bp::dict readFileOpt(std::string filename, const bp::dict& opt)
{
	// Load Options
	OctData::FileReadOptions options;

	ParameterFromOptions pfo(opt);
	options.getSetParameter(pfo);

	return readFile(filename, options);
}

bp::dict readFileDefault(std::string filename)
{
	return readFile(filename, OctData::FileReadOptions());
}

bp::dict getDefaultConfig()
{
	OctData::FileReadOptions opt;
	ParameterToOptions pto;
	opt.getSetParameter(pto);
	return pto.getValueDict();
}


using namespace boost::python;

BOOST_PYTHON_MODULE(octdata4python)
{
	bn::initialize();

	def("readFile", readFileDefault);
	def("readFile", readFileOpt);
	def("getDefaultConfig", getDefaultConfig);
};


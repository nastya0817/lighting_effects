#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <GL/glew.h>      
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>    
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <queue>
#include <random>
#include <regex>

using namespace std;

const float EPSILON = 1e-6f;
float randomFloat() {
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



struct ray {
	glm::vec3 origin;    
	glm::vec3 direction;  
	ray() = default;
	ray(glm::vec3 origin, glm::vec3 direction) :
		origin(origin)
		, direction(direction)
	{}
};

struct camera {
	glm::vec3 position;
	camera() = default;
	camera(glm::vec3 position) : position(position)
	{}
};

struct vertex {
	glm::vec3 position; 
	glm::vec3 normal;   
	glm::vec2 texCoord; 
};

struct face {
	std::vector<int> vertexIndices; 
	face() = default;
	face(std::vector<int> vertexIndices) : vertexIndices(vertexIndices)
	{}
};



struct photon;
struct scene;
struct intersection_info;
struct light {
	glm::vec3 color; 
	float intens; 

	light(const glm::vec3& color, float intensity) : color(color), intens(intensity) {}


	virtual glm::vec3 random_point() const {
		return glm::vec3(0.0f); 
	}

	virtual photon emit_photons(int num_photons) const = 0;

	virtual glm::vec3 computeDirectLight(const scene& scene, intersection_info& info) const = 0;
	virtual ~light() = default;
};




struct material {
	glm::vec3 color; 
	float reflect_coef;
	float refract_coef;
	float refract_ind;
	float shine;
	material() = default;
	material(glm::vec3 color) :
		color(color)
		, reflect_coef(0.0f)
		, refract_coef(0.0f)
		, refract_ind(0.0f)
		, shine(0.0f)
	{}
};

struct object;

struct intersection_info {
	const object* object; 
	float distance;       
	bool is_intersect; 
	glm::vec3 normal;
	glm::vec3 hit_point;
	intersection_info() : object(nullptr), distance(std::numeric_limits<float>::max()), is_intersect(false) {}
};

struct object {
	std::vector<glm::vec3> vertices; 
	std::vector<face> faces;     
	material material;
	object() = default;
	object(vector<glm::vec3> vertices, vector<face> faces, ::material material) :
		vertices(vertices)
		, faces(faces)
		, material(material)
	{}
	bool intersect_triangle(const ray& ray, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, float& t, glm::vec3& normal) const {
		glm::vec3 e1 = v1 - v0;
		glm::vec3 e2 = v2 - v0;
		glm::vec3 prec = glm::cross(ray.direction, e2);
		float det = glm::dot(e1, prec);

		if (det > -EPSILON && det < EPSILON)
			return false; 

		float f = 1.0f / det;
		glm::vec3 tvec = ray.origin - v0;
		float u = f * glm::dot(tvec, prec);

		if (u < 0.0f || u > 1.0f)
			return false;

		glm::vec3 qvec = glm::cross(tvec, e1);
		float v = f * glm::dot(ray.direction, qvec);

		if (v < 0.0f || u + v > 1.0f)
			return false;

		t = f * glm::dot(e2, qvec);
		if (t <= EPSILON)
			return false;
		normal = glm::normalize(glm::cross(e1, e2));
		return true;
	}


	virtual intersection_info intersect(const ray& ray) const {
		intersection_info info;

		for (const auto& face : faces) {
			const glm::vec3& v0 = vertices[face.vertexIndices[0]];
			const glm::vec3& v1 = vertices[face.vertexIndices[1]];
			const glm::vec3& v2 = vertices[face.vertexIndices[2]];

			float t;
			glm::vec3 normal;
			if (intersect_triangle(ray, v0, v1, v2, t, normal)) {
				if (t < info.distance) {
					info.distance = t;
					info.object = this;
					info.is_intersect = true;
					info.normal = normal;
					info.hit_point = ray.origin + ray.direction * t;
				}
			}
		}

		return info;
	}
};


struct sphere : public object {
	glm::vec3 center; 
	float radius;     

	sphere() = default;
	sphere(const glm::vec3& center, float radius, const ::material& material)
		: center(center), radius(radius) {
		this->material = material; 
	}

	intersection_info intersect(const ray& ray) const override {
		intersection_info info;

		glm::vec3 oc = ray.origin - center;
		float a = glm::dot(ray.direction, ray.direction);
		float b = 2.0f * glm::dot(oc, ray.direction);
		float c = glm::dot(oc, oc) - radius * radius;

		float discriminant = b * b - 4 * a * c;

		if (discriminant < 0) {
			info.is_intersect = false;
			return info;
		}

		float t1 = (-b - sqrt(discriminant)) / (2.0f * a);
		float t2 = (-b + sqrt(discriminant)) / (2.0f * a);

		float t = (t1 > 0) ? t1 : t2;

		if (t <= 0) {
			info.is_intersect = false;
			return info;
		}

		info.distance = t;
		info.is_intersect = true;
		info.object = this;
		info.hit_point = ray.origin + ray.direction * t;
		info.normal = glm::normalize(info.hit_point - center);

		return info;
	}
};



struct scene {
	std::vector<object*> objects; 
	std::vector<std::shared_ptr<light>> lights;
	camera camera;
	intersection_info find_closest_intersection(ray& ray) const {
		intersection_info closest_info;

		for (const auto& obj : objects) {
			intersection_info info = obj->intersect(ray);

			if (info.is_intersect && info.distance < closest_info.distance) {
				closest_info = info;
			}
		}

		return closest_info;
	}
};


struct photon : ray {
	glm::vec3 energy;
	photon() = default;
	photon(glm::vec3 origin, glm::vec3 direction, glm::vec3 energy) :
		ray(origin, direction)
		, energy(energy)
	{}

};







void createOrthogonalFrame(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent) {
	if (abs(normal.x) > abs(normal.y)) {
		tangent = glm::vec3(normal.z, 0, -normal.x);
	}
	else {
		tangent = glm::vec3(0, -normal.z, normal.y);
	}
	tangent = glm::normalize(tangent);
	bitangent = glm::normalize(glm::cross(normal, tangent));
}

glm::vec3 sampleCosineWeightedHemisphere(const glm::vec3& normal) {
	float u1 = randomFloat();
	float u2 = randomFloat();

	float r = sqrt(u1);
	float theta = 2.0f * M_PI * u2;
	glm::vec3 localVector(
		r * cos(theta),
		r * sin(theta),
		sqrt(1.0f - u1)
	);
	glm::vec3 tangent, bitangent;
	createOrthogonalFrame(normal, tangent, bitangent);

	return localVector.x * tangent + localVector.y * bitangent + localVector.z * normal;
}

thread_local std::random_device rd;
thread_local std::mt19937 gen(rd());

float get_random(float a, float b) {
	std::uniform_real_distribution<float> dis(a, b);
	return dis(gen);
}

ray reflect_spherical(const glm::vec3& from, const glm::vec3& normal)  {
	glm::vec3 new_dir;
	do {
		float x1, x2;
		float sqrx1, sqrx2;
		do {
			x1 = get_random(-1.f, 1.f);
			x2 = get_random(-1.f, 1.f);
			sqrx1 = x1 * x1;
			sqrx2 = x2 * x2;
		} while (sqrx1 + sqrx2 >= 1);
		float fx1x2 = std::sqrt(1.f - sqrx1 - sqrx2);
		new_dir.x = 2.f * x1 * fx1x2;
		new_dir.y = 2.f * x2 * fx1x2;
		new_dir.z = 1.f - 2.f * (sqrx1 + sqrx2);
	} while (glm::dot(new_dir, normal) < 0);
	ray res;
	res.direction = glm::normalize(new_dir);
	res.origin = from + 0.0001f * res.direction;
	return res;
}

vector<photon> photons;
vector<photon> caustic;
vector<photon> direct_photons;

//void trace_photon(const scene& scene, photon& photon, std:: string& path, int bounce = 0) {
//	const int maxBounces = 4; // Максимальное число отражений/преломлений
//
//	// Если достигнут лимит отражений, прекращаем трассировку
//	if (bounce >= maxBounces) {
//		return;
//	}
//
//	// Создаем луч для текущего фотона
//	ray photonRay(photon.origin, photon.direction);
//
//	// Ищем ближайшее пересечение луча с объектами сцены
//	intersection_info info = scene.find_closest_intersection(photonRay);
//
//	// Если пересечение не найдено, прекращаем трассировку
//	if (!info.is_intersect) {
//		return;
//	}
//
//
//	// Обновляем позицию фотона
//	photon.origin = info.hit_point;
//	
//	// Если энергия фотона стала слишком малой, прекращаем трассировку
//	if (glm::length(photon.energy) < 0.00001f) {
//		return;
//	}
//
//	// Определяем тип взаимодействия
//	char interaction_type = '\0';
//	if (info.object->material.reflect_coef > 0.0f) {
//		interaction_type = 'S'; // Зеркальное отражение
//	}
//	else if (info.object->material.refract_coef > 0.0f) {
//		interaction_type = 'R'; // Преломление
//	}
//	else {
//		interaction_type = 'D'; // Диффузное отражение
//	}
//
//
//	// Обновляем путь фотона
//	path += interaction_type;
//
//	// Проверяем условия для каустической карты
//	if (std::regex_match(path, std::regex("L(S|R)+D")) || std::regex_match(path, std::regex("L(S|R)+A"))) {
//		caustic.push_back(photon);
//	}
//	// Проверяем условия для глобальной карты
//	else if (std::regex_match(path, std::regex("L(S|D|R)*D")) || std::regex_match(path, std::regex("L(S|D|R)*A"))) {
//		photons.push_back(photon);
//	}
//
//	//зеркальная
//	if (interaction_type == 'S') {
//		// Уменьшаем энергию фотона на коэффициент отражения
//		photon.energy *= info.object->material.reflect_coef;
//
//		// Вычисляем отраженное направление
//		glm::vec3 reflected_direction = glm::reflect(photon.direction, info.normal);
//		photon.direction = reflected_direction;
//
//		// Продолжаем трассировку
//		trace_photon(scene, photon, path, bounce + 1);
//	}
//	//прозрачная
//	else if (interaction_type == 'R') {
//		// Уменьшаем энергию фотона на коэффициент преломления
//		//photon.energy *= info.object->material.refract_coef;
//
//		auto normal = info.normal;
//
//		float n1 = 1.0f; // Показатель преломления внешней среды (воздух)
//		float n2 = info.object->material.refract_ind; // Показатель преломления объекта
//		glm::vec3 view_dir = glm::normalize(photon.direction); // Направление к камере
//		float cos_theta = glm::dot(normal, -view_dir);
//
//		// Если луч идет изнутри объекта, меняем нормаль и показатели преломления
//		if (cos_theta < 0) {
//			normal = -normal;
//			cos_theta = -cos_theta;
//			std::swap(n1, n2); // Меняем местами n1 и n2
//		}
//
//		float refr_ratio = n1 / n2;
//		float sin2_theta_t = refr_ratio * refr_ratio * (1.0f - cos_theta * cos_theta);
//
//		if (sin2_theta_t <= 1.0f) {
//			// Преломление возможно
//			float cos_theta_t = sqrt(1.0f - sin2_theta_t);
//			glm::vec3 refract_dir = refr_ratio * view_dir + (refr_ratio * cos_theta - cos_theta_t) * normal;
//			refract_dir = glm::normalize(refract_dir);
//			photon.direction = refract_dir;
//			// Продолжаем трассировку
//			trace_photon(scene, photon, path, bounce + 1);
//		}
//
//		
//	}
//	//диффузная
//	else {
//		// Обновляем энергию фотона (умножаем на альбедо поверхности)
//		//photons.push_back(photon);
//		photon.energy *= info.object->material.color;
//		photon.direction = sampleCosineWeightedHemisphere(info.normal);
//		trace_photon(scene, photon, path, bounce + 1);
//
//	}
//
//}
//




float FresnelSchlick(float cosNL, float n1, float n2) {
	float f0 = std::pow((n1 - n2) / (n1 + n2), 2);
	float absCosNL = std::abs(cosNL);
	return f0 + (1.f - f0) * std::pow(1.f - absCosNL, 5.f);
}


bool can_refract(const intersection_info& info, float cosNL, bool in_object) {
	auto& mat = info.object->material;
	if (mat.refract_ind == 1.0f) return false; 

	float n1 = 1.0f; 
	float n2 = mat.refract_ind;
	if (in_object) std::swap(n1, n2);

	float sinTheta2 = (n1 / n2) * std::sqrt(std::max(1.0f - cosNL * cosNL, 0.f));
	if (sinTheta2 >= 1.0f) return false;
	float criticalAngleFactor = std::min(1.0f, sinTheta2 * 2.0f);
	float refr_probability = (1.f - FresnelSchlick(cosNL, n1, n2)) * criticalAngleFactor;

	return get_random(0.f, 1.f) <= refr_probability;
}

char get_type(const intersection_info& info, float cosNL, glm::vec3& energy, bool in_obj) {
	if (can_refract(info, cosNL, in_obj))
		return 'R';
	float e = get_random(0.0f, 2.0f);
	material mat = info.object->material;
	float max_en = max(max(energy.r, energy.g), energy.b);
	glm::vec3 d = mat.color * energy;
	float max_d = max(max(d.r, d.g), d.b);
	float pd = max_d / max_en;
	if (e <= pd) {
		if (info.object->material.reflect_coef > 0.8f) {
			return 'S';
		}
		return 'D';
	}
	glm::vec3 s = mat.reflect_coef * energy;
	float max_s= max(max(s.r, s.g), s.b);
	float ps = max_s / max_en;
	if (e <= pd + ps)
		return 'S';
	return 'A';
}


const int max_bounce = 5;
void trace_photon(const scene& scene, photon& photon, std::string& path, bool in_object, int bounce = 0) {
	ray photonRay(photon.origin, photon.direction);
	intersection_info info = scene.find_closest_intersection(photonRay);

	if (!info.is_intersect) {
		return;
	}

	float cosNL = glm::dot(-normalize(photon.direction), normalize(info.normal));
	photon.origin = info.hit_point;

	if (info.object->material.reflect_coef == 1)
		int a = 1;

	char interaction_type = get_type(info, cosNL, photon.energy, in_object);
	::photon new_photon=photon;
	switch (interaction_type)
	{
	case 'R': 
	{
		if (glm::dot(photon.direction, info.normal) > 0) {
			info.normal = -info.normal;
		}
		float refr1 = 1.0;
		float refr2 = info.object->material.refract_ind;
		if (in_object)
			swap(refr1, refr2);
		float eta = refr1 / refr2;
		float c1 = -glm::dot(photon.direction, info.normal);
		float w = eta * c1;
		float c2m = (w - eta) * (w + eta);
		if (c2m < -1.f) {
			return;
		}
		new_photon.direction = eta * photon.direction + (w - sqrt(1.f + c2m)) * info.normal;
		new_photon.origin = info.hit_point + 0.01f * -info.normal;
		in_object = !in_object;
		break;
	}
	case 'S':
	{
		auto dnd = glm::dot(photon.direction, info.normal);
		glm::vec3 refl_dir = photon.direction - 2.f * info.normal * dnd;
		new_photon.direction = glm::normalize(refl_dir);
		new_photon.origin = info.hit_point + 0.01f * info.normal;
		break;
	}
	case 'D':
	{
		if (info.object->material.reflect_coef > 0) {
			int a = 0;
		}
		ray r = reflect_spherical(info.hit_point, info.normal);
		new_photon.direction = r.direction;
		new_photon.origin = r.origin;
		new_photon.energy *= info.object->material.color;
		break;

	}
	case 'A':
	{
		if (info.object->material.reflect_coef > 0) {
			int a = 0;
		}
		break;
	}
	default:
		break;
	}
	path += interaction_type;
	if (interaction_type == 'A' || interaction_type == 'D') {
		if (std::regex_match(path, std::regex("L(R)+D")) || std::regex_match(path, std::regex("L(R)+A"))) {
			caustic.push_back(photon);
		}
		else if (std::regex_match(path, std::regex("L(S|D|R)+(D|A)"))) {
			photons.push_back(photon);
		}
		else if (std::regex_match(path, std::regex("L(D|A)")))
			direct_photons.push_back(photon);
	}
	if (interaction_type!='A')
		trace_photon(scene, new_photon, path, in_object, bounce + 1);
	
}


void trace_photons(const scene& scene, int num_photons, int coef=1) {
	int num_for_light = num_photons / scene.lights.size();
	for (auto& light : scene.lights) {
		for (int i = 0; i < num_for_light; ++i) {
			photon p = light->emit_photons(num_for_light);
			p.energy *= coef;
			string path = "L";
			trace_photon(scene, p, path, false);
		}
	}
	cout << photons.size();
}

struct point_light : public light {
	glm::vec3 position; 

	point_light(const glm::vec3& position, const glm::vec3& color, float intensity)
		: light(color, intensity), position(position) {}

	
	photon emit_photons(int num_photons) const override {
		photon photon;
		photon.origin = position;

		glm::vec3 light_power = glm::vec3(this->color * this->intens);
		float theta = 2.0f * M_PI * randomFloat();
		float phi = M_PI * randomFloat(); 
		glm::vec3 direction = glm::vec3(
			sin(phi) * cos(theta),
			sin(phi) * sin(theta),
			cos(phi)
		);
		photon.direction = direction;
		photon.energy = light_power / static_cast<float>(num_photons);
		return photon;
	}

	glm::vec3 computeDirectLight(const scene& scene, intersection_info& info) const override {
		glm::vec3 ambientColor(0.0f);  
		glm::vec3 diffuseColor(0.0f);  
		ambientColor = info.object->material.color * 0.1f;
		

		glm::vec3 lightDir = this->position - info.hit_point;
		float distanceToLight = glm::length(lightDir);
		lightDir = glm::normalize(lightDir);

		ray shadowRay(info.hit_point + info.normal * 0.001f, lightDir); 
		intersection_info shadowInfo = scene.find_closest_intersection(shadowRay);
		if (!shadowInfo.is_intersect || shadowInfo.distance > distanceToLight) {
			float diffuseIntensity = glm::max(glm::dot(info.normal, lightDir), 0.0f);
			diffuseColor += this->color * info.object->material.color * diffuseIntensity * this->intens;
			float attenuation = 1.0f / (1.0f + 0.1f * distanceToLight + 0.01f * distanceToLight * distanceToLight);
			diffuseColor *= attenuation;
		}
		glm::vec3 finalColor = ambientColor + diffuseColor;
		finalColor = glm::clamp(finalColor, 0.0f, 1.0f);

		return finalColor;
	}


};



struct square_light : public light {
	glm::vec3 position;  
	glm::vec3 normal;   
	glm::vec3 tangent;   
	float width;        
	float height;       

	square_light(const glm::vec3& position, const glm::vec3& normal, float width, float height, const glm::vec3& color, float intensity)
		: light(color, intensity), position(position), normal(glm::normalize(normal)), width(width), height(height) {
		if (std::abs(normal.y) > 0.999f) {
			tangent = glm::normalize(glm::cross(normal, glm::vec3(1, 0, 0)));
		}
		else {
			tangent = glm::normalize(glm::cross(normal, glm::vec3(0, 1, 0)));
		}

		if (std::abs(glm::dot(tangent, normal)) > 0.001f) {
			throw std::runtime_error("Tangent is not perpendicular to normal!");
		}
	}

	glm::vec3 random_point() const override {
		float u = randomFloat() - 0.5f; 
		float v = randomFloat() - 0.5f; 
		return position + u * height * tangent + v * width * glm::cross(normal, tangent);
	}

	photon emit_photons(int num_photons) const override {
		glm::vec3 start_point = random_point();

		glm::vec3 light_power = glm::vec3(this->color * this->intens * this->width * this->height);

		float theta = 2.0f * M_PI * randomFloat(); 
		float phi = M_PI * 0.5f * randomFloat();   
		glm::vec3 localDirection = glm::vec3(
			sin(phi) * cos(theta),
			sin(phi) * sin(theta),
			cos(phi)
		);

		glm::vec3 bitangent = glm::cross(normal, tangent);
		glm::vec3 worldDirection = localDirection.x * tangent +
			localDirection.y * bitangent +
			localDirection.z * normal;

		worldDirection = glm::normalize(worldDirection);

		photon p;
		p.origin = start_point;
		p.direction = worldDirection;
		p.energy = light_power / static_cast<float>(num_photons); // Энергия фотона
		return p;
	}


	bool is_on_light(const glm::vec3& point, float epsilon = 1e-5f) const {
		glm::vec3 to_point = point - position;
		float distance_to_plane = glm::dot(to_point, normal);

		if (std::abs(distance_to_plane) > epsilon) {
			return false;
		}

		glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

		float u = glm::dot(to_point, tangent);    
		float v = glm::dot(to_point, bitangent);  

		bool inside_width = (v >= -width * 0.5f - epsilon) && (v <= width * 0.5f + epsilon);
		bool inside_height = (u >= -height * 0.5f - epsilon) && (u <= height * 0.5f + epsilon);

		return inside_width && inside_height;
	}

	glm::vec3 computeDirectLight(const scene& scene, intersection_info& info) const override {
		glm::vec3 ambientColor(0.0f);  
		glm::vec3 diffuseColor(0.0f);  

		ambientColor = info.object->material.color * 0.1f; 


		if (is_on_light(info.hit_point, 0.01)) {
			return color;
		}


		glm::vec3 totalDiffuse(0.0f);
		int numSamples = 300; 

		for (int i = 0; i < numSamples; ++i) {
			glm::vec3 lightPoint = this->random_point();
			glm::vec3 lightDir = lightPoint - info.hit_point;
			float distanceToLight = glm::length(lightDir);
			lightDir = glm::normalize(lightDir);

			float NdotL = glm::max(glm::dot(info.normal, lightDir), 0.0f);
			if (NdotL <= 0.0f) continue; 

			float lightCos = glm::max(glm::dot(-lightDir, this->normal), 0.0f);
			if (lightCos <= 0.0f) continue; 

			ray shadowRay(info.hit_point + info.normal * 0.001f, lightDir);
			intersection_info shadowInfo = scene.find_closest_intersection(shadowRay);

			if (!shadowInfo.is_intersect || shadowInfo.distance > distanceToLight) {
				float attenuation = 1.0f / (1.0f + 0.1f * distanceToLight + 0.01f * distanceToLight * distanceToLight);
				float area = this->width * this->height;
				glm::vec3 lightContrib = this->color * info.object->material.color
					* NdotL * lightCos * area * this->intens
					* attenuation / (distanceToLight * distanceToLight);

				totalDiffuse += lightContrib;
			}
		}

		diffuseColor += totalDiffuse / static_cast<float>(numSamples);
		glm::vec3 finalColor = ambientColor + diffuseColor;
		finalColor = glm::clamp(finalColor, 0.0f, 1.0f);

		return finalColor;
	}

};

void ParseObj(const std::string& filename, std::vector<object>& objects) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open .obj file: " + filename);
	}

	std::vector<glm::vec3> temp_positions;  
	std::vector<glm::vec2> temp_texCoords;
	std::vector<glm::vec3> temp_normals;   

	object currentObject; 
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream lineStream(line);
		std::string prefix;
		lineStream >> prefix;

		if (prefix == "o") {
			if (!currentObject.vertices.empty()) {
				objects.push_back(currentObject);
				currentObject = object(); 
			}
		}
		else if (prefix == "v") {
			glm::vec3 position;
			lineStream >> position.x >> position.y >> position.z;
			temp_positions.push_back(position);
		}
		else if (prefix == "vt") { 
			glm::vec2 texCoord;
			lineStream >> texCoord.x >> texCoord.y;
			texCoord.y = 1.0f - texCoord.y; 
			temp_texCoords.push_back(texCoord);
		}
		else if (prefix == "vn") {
			glm::vec3 normal;
			lineStream >> normal.x >> normal.y >> normal.z;
			temp_normals.push_back(normal);
		}
		else if (prefix == "f") { 
			face face;
			std::string vertexInfo;
			while (lineStream >> vertexInfo) {
				size_t pos1 = vertexInfo.find('/');
				size_t pos2 = vertexInfo.find('/', pos1 + 1);

				int vertexIndex = std::stoi(vertexInfo.substr(0, pos1)) - 1;
				int texCoordIndex = (pos2 > pos1 + 1) ? std::stoi(vertexInfo.substr(pos1 + 1, pos2 - pos1 - 1)) - 1 : -1;
				int normalIndex = (pos2 != std::string::npos) ? std::stoi(vertexInfo.substr(pos2 + 1)) - 1 : -1;

				vertex vertex;
				vertex.position = temp_positions[vertexIndex];
				vertex.texCoord = (texCoordIndex >= 0) ? temp_texCoords[texCoordIndex] : glm::vec2(0.0f, 0.0f);
				vertex.normal = (normalIndex >= 0) ? temp_normals[normalIndex] : glm::vec3(0.0f, 0.0f, 0.0f);

				currentObject.vertices.push_back(vertex.position);
				face.vertexIndices.push_back(currentObject.vertices.size() - 1);
			}

			currentObject.faces.push_back(face);
		}
	}
	if (!currentObject.vertices.empty()) {
		objects.push_back(currentObject);
	}
}



//
//glm::vec3 computeDirectLight(const scene& scene, const intersection_info& info) {
//	glm::vec3 ambientColor(0.0f);  // Фоновое освещение
//	glm::vec3 diffuseColor(0.0f);  // Диффузное освещение
//
//	// Ambient освещение (постоянное для всех точек)
//	ambientColor = info.object->material.color * 0.1f; // 10% ambient
//
//	// Проходим по всем источникам света в сцене
//	for (const auto& slight : scene.lights) {
//		if (slight->getType() == LightType::Point) {
//			// Вектор от точки пересечения к источнику света
//			auto light = dynamic_cast<point_light*>(slight.get());
//			glm::vec3 lightDir = light->position - info.hit_point;
//			float distanceToLight = glm::length(lightDir);
//			lightDir = glm::normalize(lightDir);
//
//			// Проверяем, не находится ли точка в тени
//			ray shadowRay(info.hit_point + info.normal * 0.001f, lightDir); // Смещаем начало луча, чтобы избежать самопересечения
//			intersection_info shadowInfo = scene.find_closest_intersection(shadowRay);
//
//			// Если луч до источника света не перекрыт, добавляем вклад света
//			if (!shadowInfo.is_intersect || shadowInfo.distance > distanceToLight) {
//				// Диффузное освещение
//				float diffuseIntensity = glm::max(glm::dot(info.normal, lightDir), 0.0f);
//				diffuseColor += light->color * info.object->material.color * diffuseIntensity * light->intens;
//
//				// Учитываем затухание света
//				float attenuation = 1.0f / (1.0f + 0.1f * distanceToLight + 0.01f * distanceToLight * distanceToLight);
//
//				// Применяем затухание к диффузному освещению
//				diffuseColor *= attenuation;
//			}
//		}
//		else if (slight->getType() == LightType::Square) {
//			auto squareLight = dynamic_cast<square_light*>(slight.get());
//			if (!squareLight) continue;
//
//			// Проверяем, находится ли точка пересечения на источнике света
//			if (isPointOnLight(info.hit_point, squareLight)) {
//				return glm::vec3(1.0, 1.0, 1.0); // Пропускаем расчёт освещения для этой точки
//			}
//
//			glm::vec3 totalDiffuse(0.0f);
//			int numSamples = 100; // Увеличиваем количество сэмплов для более гладких теней
//
//			for (int i = 0; i < numSamples; ++i) {
//				// Сэмплируем случайную точку на источнике света
//				glm::vec3 lightPoint = squareLight->random_point();
//				glm::vec3 lightDir = lightPoint - info.hit_point;
//				float distanceToLight = glm::length(lightDir);
//				lightDir = glm::normalize(lightDir);
//
//				// Косинус угла между нормалью поверхности и направлением на свет
//				float NdotL = glm::max(glm::dot(info.normal, lightDir), 0.0f);
//				if (NdotL <= 0.0f) continue; // Пропускаем, если свет с обратной стороны
//
//				// Косинус угла между нормалью источника и направлением на точку (закон Ламберта)
//				float lightCos = glm::max(glm::dot(-lightDir, squareLight->normal), 0.0f);
//				if (lightCos <= 0.0f) continue; // Пропускаем, если точка не видна с источника
//
//				// Проверка тени
//				ray shadowRay(info.hit_point + info.normal * 0.001f, lightDir);
//				intersection_info shadowInfo = scene.find_closest_intersection(shadowRay);
//
//				if (!shadowInfo.is_intersect || shadowInfo.distance > distanceToLight) {
//					// Вычисляем вклад света
//					float attenuation = 1.0f / (1.0f + 0.1f * distanceToLight + 0.01f * distanceToLight * distanceToLight);
//					float area = squareLight->width * squareLight->height;
//
//					// Формула для площадиного источника света:
//					// (intensity * color * NdotL * lightCos * area) / (distance² * numSamples)
//					glm::vec3 lightContrib = squareLight->color * info.object->material.color
//						* NdotL * lightCos * area * squareLight->intens
//						* attenuation / (distanceToLight * distanceToLight);
//
//					totalDiffuse += lightContrib;
//				}
//			}
//
//			// Усредняем результат
//			diffuseColor += totalDiffuse / static_cast<float>(numSamples);
//		}
//	}
//
//
//	// Итоговый цвет = ambient + diffuse
//	glm::vec3 finalColor = ambientColor + diffuseColor;
//
//	// Ограничиваем цвет в диапазоне [0, 1]
//	finalColor = glm::clamp(finalColor, 0.0f, 1.0f);
//
//	return finalColor;
//}
//


////////
struct kd_node {
	photon photon;
	kd_node* left;
	kd_node* right;
	int axis;

	kd_node(const ::photon& p, int ax): 
		photon(p)
		, left(nullptr)
		, right(nullptr)
		, axis(ax)
	{}

	~kd_node() {
		if (left != nullptr) {
			delete left;
		}

		if (right != nullptr) {
			delete right;
		}
	}
};



kd_node* build_kd_tree(vector<photon>& photons, int depth = 0) {
	if (photons.empty()) return nullptr;
	int ax = depth % 3;

	std::sort(photons.begin(), photons.end(), [ax](const photon& a, const photon& b) {
		return a.origin[ax] < b.origin[ax];
	});

	size_t median = photons.size() / 2;
	kd_node* node = new kd_node(photons[median], ax);

	std::vector<photon> left(photons.begin(), photons.begin() + median);
	std::vector<photon> right(photons.begin() + median + 1, photons.end());
	node->left = build_kd_tree(left, depth + 1);
	node->right = build_kd_tree(right, depth + 1);

	return node;
}


struct photon_dist {
	photon p;
	float dist;

	bool operator<(const photon_dist& other) const{
		return dist < other.dist;
	}
};

float dist_photon(const glm::vec3 a, const glm::vec3 b) {
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	float dz = a[2] - b[2];
	return dx * dx + dy * dy + dz * dz;
}

void find_k_near(kd_node* root, glm::vec3 target, std::priority_queue<photon_dist>& q, int k) {
	if (!root) return;

	int ax = root->axis;
	float dist = dist_photon(target, root->photon.origin);

	if (q.size() < k || dist < q.top().dist) {
		q.push({ root->photon, dist });
		if (q.size() > k) {
			q.pop(); 
		}
	}

	kd_node* near_tree;
	kd_node* far_tree;
	float a = target[ax] - root->photon.origin[ax];
	if (a < 0) {
		near_tree = root->left;
		far_tree = root->right;
	}
	else {
		near_tree = root->right;
		far_tree = root->left;
	}

	find_k_near(near_tree, target, q, k);

	if (q.size() < k || a * a < q.top().dist) {
		find_k_near(far_tree, target, q, k);
	}
}

vector<photon_dist> find_k_nearest_photons(kd_node* root, glm::vec3 target, int k) {
	std::priority_queue<photon_dist> q;
	find_k_near(root, target, q, k);
	std::vector<photon_dist> result;
	while (!q.empty()) {
		result.push_back(q.top());
		q.pop();
	}
	return result;
}








//glm::vec3 computeIndirectLight(kd_node* root, intersection_info& info, int k) {
//	if (!root) {
//		return glm::vec3(0.0f); // Если фотонов нет, возвращаем черный цвет
//	}
//
//	// Находим k ближайших фотонов
//	std::vector<photon_dist> nearest_photons = find_k_nearest_photons(root, info.hit_point, k);
//
//	/*if (nearest_photons[0].dist >= 0.5)
//		return glm::vec3(0);*/
//
//	float a = nearest_photons[0].dist*M_PI*M_PI;
//	glm::vec3 total_energy(0.0f);
//	for (const auto& photon : nearest_photons) {
//		if (dot(-photon.p.direction, info.normal)>0)
//		total_energy += photon.p.energy * dot(-photon.p.direction,info.normal) * info.object->material.color;
//	}
//	total_energy /= a;
//
//
//	// Параметр сглаживания (bandwidth)
//	//float h = 1.0f; // Может быть настроен экспериментально
//	//h = sqrt(nearest_photons[0].dist);
//	//// Суммарный вес и энергия
//	//float total_weight = 0.0f;
//	//glm::vec3 total_energy(0.0f);
//
//	//// Вычисляем веса и энергию
//	//for (const auto& photon : nearest_photons) {
//	//	float distance_squared = photon.dist; // Расстояние уже вычислено в find_k_nearest_photons
//	//	float weight = exp(-distance_squared / (2 * h)); // Гауссово ядро
//	//	total_weight += weight;
//	//	total_energy += weight * photon.p.energy; // Энергия фотона (цвет)
//	//}
//
//	//// Нормализуем энергию
//	//if (total_weight > 0.0f) {
//	//	total_energy /= total_weight;
//	//}
//
//	return total_energy;
//}

//work

//glm::vec3 computeIndirectLight(kd_node* root, intersection_info& info, int k) {
//	if (!root) {
//		return glm::vec3(0.0f);
//	}
//
//	// Получаем нормаль поверхности в точке пересечения
//	glm::vec3 plane_normal = info.normal;
//	glm::vec3 plane_point = info.hit_point;
//
//	// Находим k ближайших фотонов
//	std::vector<photon_dist> nearest_photons = find_k_nearest_photons(root, plane_point, k);
//	if (info.normal.r == 0.25)
//		return glm::vec3(0);
//
//	// Фильтруем фотоны: оставляем только те, которые лежат на плоскости
//	std::vector<photon_dist> filtered_photons;
//	for (const auto& photon : nearest_photons) {
//		// Проверяем, лежит ли фотон на плоскости
//		float distance_to_plane = glm::dot(photon.p.origin - plane_point, plane_normal);
//		if (std::abs(distance_to_plane) < 0.001f) { // Порог можно настроить
//			filtered_photons.push_back(photon);
//		}
//	}
//
//	// Если не осталось фотонов, возвращаем черный цвет
//	if (filtered_photons.empty()) {
//		return glm::vec3(0.0f);
//	}
//
//	// Максимальное расстояние (квадрат радиуса сферы сбора)
//	float max_distance_sq = filtered_photons[0].dist;
//
//
//	// Константа для гауссова фильтра (1/(2*sigma^2)), где sigma = radius/3
//	// Можно регулировать коэффициент 3.0 для изменения формы фильтра
//	float gaussian_alpha = 1.0f / (2.0f * max_distance_sq / 2.0f);
//
//	glm::vec3 total_energy(0.0f);
//	float total_weight = 0.0f;
//
//	for (const auto& photon : filtered_photons) {
//		// Нормализованное квадратичное расстояние [0..1]
//		float normalized_dist_sq = photon.dist / max_distance_sq;
//
//		// Гауссов вес (exp(-d^2/(2*sigma^2)))
//		float weight = exp(-gaussian_alpha * photon.dist);
//
//		total_energy += photon.p.energy * weight;
//		total_weight += weight;
//	}
//
//	// Нормализуем по площади (πR²) и суммарному весу
//	float area = max_distance_sq * M_PI * M_PI;
//	if (total_weight > 0.0f) {
//		total_energy /= (area );
//	}
//
//	return total_energy;
//}


glm::vec3 computeIndirectLight(kd_node* root, intersection_info& info, int k) {
	if (!root) {
		return glm::vec3(0.0f);
	}

	glm::vec3 plane_normal = info.normal;
	glm::vec3 plane_point = info.hit_point;

	std::vector<photon_dist> nearest_photons = find_k_nearest_photons(root, plane_point, k);
	if (info.normal.r == 0.25) 
		return glm::vec3(0);
	std::vector<photon_dist> filtered_photons;
	for (const auto& photon : nearest_photons) {
		float distance_to_plane = glm::dot(photon.p.origin - plane_point, plane_normal);
		if (std::abs(distance_to_plane) < 0.001f) { 
			filtered_photons.push_back(photon);
		}
	}

	if (filtered_photons.empty()) {
		return glm::vec3(0.0f);
	}

	float a = filtered_photons[0].dist * M_PI * M_PI;
	glm::vec3 total_energy(0.0f);
	for (const auto& photon : filtered_photons) {
		//if (dot(-photon.p.direction, info.normal) > 0)
		total_energy += photon.p.energy/** dot(-photon.p.direction, info.normal)*/ /** info.object->material.color*/;
	}
	total_energy /= a;

	return total_energy;
}

float fresnelSchlick(float cosTheta, float n1, float n2) {
	float R0 = pow((n1 - n2) / (n1 + n2), 2.0);
	return R0 + (1.0f - R0) * pow(1.0f - cosTheta, 5.0f);
}

const int k = 300;
const int MAX_DEPTH = 5;


glm::vec3 trace_ray(kd_node* global_map, kd_node* direct_map, kd_node* caustic_map, const scene& scene, ray& ray, const std::vector<photon>& photons, int depth = 0) {
	if (depth > MAX_DEPTH) {
		return glm::vec3(0.0f);
	}

	intersection_info info = scene.find_closest_intersection(ray);

	if (info.is_intersect) {
		glm::vec3 base_color = info.object->material.color;
		glm::vec3 directLight(0.0f);

		//directLight = computeIndirectLight(direct_map, info, k);
		//directLight *= 3;
		for (auto& light : scene.lights) {
			//directLight += light->computeDirectLight(scene, info);
		}
		glm::vec3 indirectLight(0);
		//if (info.object->material.refract_coef!=1.0 && info.object->material.reflect_coef==0.0)
		//indirectLight = computeIndirectLight(global_map, info, k);	//cout << 1;


		//indirectLight *= 3;
		//indirectLight *= info.object->material.color;


		glm::vec3 causticLight(0);
		causticLight = computeIndirectLight(caustic_map, info, 50);
		//causticLight *= 3;
		//causticLight *= info.object->material.color;

		glm::vec3 reflectionColor(0.0f);
		glm::vec3 refractionColor(0.0f);

		float reflect_coef = info.object->material.reflect_coef;
		float refract_coef = info.object->material.refract_coef;
		glm::vec3 normal = glm::normalize(info.normal);
		float eps = 1e-4f;

		bool inside = false;
		float n1 = 1.0f; 
		float n2 = info.object->material.refract_ind;
		float n = n1 / n2;

		glm::vec3 view_dir = glm::normalize(ray.direction);
		float cos_theta = glm::dot(view_dir, normal);

		if (cos_theta > 0) {
			inside = true;
			normal = -normal;
			std::swap(n1, n2);
			n = n1 / n2;
		}
		else {
			cos_theta = -cos_theta;
		}
		float R0 = pow((n1 - n2) / (n1 + n2), 2);
		float fresnel = R0 + (1 - R0) * pow(1 - cos_theta, 5);
		if (reflect_coef > 0.0f || refract_coef > 0.0f) {
			glm::vec3 reflect_dir = glm::reflect(view_dir, normal);
			::ray reflected_ray(info.hit_point + reflect_dir * eps, reflect_dir);
			reflectionColor = trace_ray(global_map, direct_map, caustic_map, scene, reflected_ray, photons, depth + 1);
		}

		if (refract_coef > 0.0f) {
			float sin2_theta_t = n * n * (1.0f - cos_theta * cos_theta);

			if (sin2_theta_t < 1.0f) {
				float cos_theta_t = sqrt(1.0f - sin2_theta_t);
				glm::vec3 refract_dir = n * view_dir + (n * cos_theta - cos_theta_t) * normal;
				refract_dir = glm::normalize(refract_dir);
				if (!glm::any(glm::isnan(refract_dir))) {
					::ray refracted_ray(info.hit_point + refract_dir * eps, refract_dir);
					refractionColor = trace_ray(global_map, direct_map, caustic_map, scene, refracted_ray, photons, depth + 1);

					if (inside) {
						float distance = info.distance;
						glm::vec3 absorption(0.01f); 
						refractionColor *= glm::exp(-absorption * distance);
					}
				}
			}
			else {
				refractionColor = reflectionColor;
				fresnel = 1.0f;
			}
		}


		glm::vec3 finalColor;
		if (refract_coef > 0.0f) {
			finalColor = fresnel * reflectionColor + (1.0f - fresnel) * refractionColor;
		}
		else {
			finalColor = (1.0f - reflect_coef - refract_coef) * (directLight + indirectLight + causticLight) * base_color +
				reflect_coef * reflectionColor +
				refract_coef * refractionColor;
		}


		return glm::min(finalColor, glm::vec3(1.0f));
	}

	return glm::vec3(0.0f); // Фон
}

//glm::vec3 trace_ray(kd_node* global_map, kd_node* direct_map, kd_node* caustic_map, const scene& scene, ray& ray, const std::vector<photon>& photons, int depth = 0) {
//	if (depth > MAX_DEPTH) {
//		return glm::vec3(0.0f); // Прерываем рекурсию
//	}
//
//	// Ищем ближайшее пересечение луча с объектами сцены
//	intersection_info info = scene.find_closest_intersection(ray);
//
//
//	if (info.is_intersect) {
//		glm::vec3 base_color = glm::vec3(0.0f, 0.0f, 0.0f);
//		// Вычисляем прямое освещение (например, от источников света)
//		glm::vec3 directLight(0);
//		
//		for (auto& light : scene.lights) {
//			directLight+=light->computeDirectLight(scene, info);
//		}
//
//
//		//glm::vec3 directLight = computeDirectLight(scene, info);
//
//		// Вычисляем непрямое освещение с использованием фотонной карты
//		//glm::vec3 indirectLight (0);
//		glm::vec3 indirectLight(0);
//		//if (info.object->material.refract_coef!=1.0 && info.object->material.reflect_coef==0.0)
//			//indirectLight = computeIndirectLight(global_map, info, k);	//cout << 1;
//		
//		//indirectLight *= 2;
//		//indirectLight*=info.object->material.color;
//		
//
//		glm::vec3 causticLight(0);
//		//causticLight = computeIndirectLight(caustic_map, info, 50);
//		//causticLight *= 3;
//		//causticLight *= info.object->material.color;
//		
//
//		// Обработка отражения
//		float reflect_coef = info.object->material.reflect_coef;
//		glm::vec3 normal = info.normal;
//		float eps = 1e-5;
//
//		// Отражение
//		glm::vec3 reflectionColor(0.0f);
//		if (reflect_coef > 0.0f) {
//			glm::vec3 reflect_dir = glm::reflect(ray.direction, normal);
//			::ray reflected_ray(info.hit_point + reflect_dir * eps, reflect_dir);
//			reflectionColor = trace_ray(global_map, direct_map, caustic_map, scene, reflected_ray, photons, depth + 1);
//		}
//
//		// Преломление
//		glm::vec3 refractionColor(0.0f);
//		float refract_coef = info.object->material.refract_coef;
//		if (refract_coef > 0.0f) {
//			float n1 = 1.0f; // Показатель преломления внешней среды (воздух)
//			float n2 = info.object->material.refract_ind; // Показатель преломления объекта
//			glm::vec3 view_dir = glm::normalize(ray.direction); // Направление к камере
//			float cos_theta = glm::dot(normal, -view_dir);
//
//			// Если луч идет изнутри объекта, меняем нормаль и показатели преломления
//			if (cos_theta < 0) {
//				normal = -normal;
//				cos_theta = -cos_theta;
//				std::swap(n1, n2); // Меняем местами n1 и n2
//			}
//
//			float refr_ratio = n1 / n2;
//			float sin2_theta_t = refr_ratio * refr_ratio * (1.0f - cos_theta * cos_theta);
//
//			if (sin2_theta_t <= 1.0f) {
//				// Преломление возможно
//				float cos_theta_t = sqrt(1.0f - sin2_theta_t);
//				glm::vec3 refract_dir = refr_ratio * view_dir + (refr_ratio * cos_theta - cos_theta_t) * normal;
//				refract_dir = glm::normalize(refract_dir);
//
//				::ray refracted_ray(info.hit_point + refract_dir * eps, refract_dir);
//				refractionColor = trace_ray(global_map, direct_map, caustic_map, scene, refracted_ray, photons, depth + 1);
//			}
//			else {
//				glm::vec3 reflect_dir = glm::reflect(ray.direction, normal);
//				::ray reflected_ray(info.hit_point + reflect_dir * eps, reflect_dir);
//				reflectionColor = trace_ray(global_map, direct_map, caustic_map, scene, reflected_ray, photons, depth + 1);
//			}
//		}
//
//		// Смешивание цветов отражения и преломления
//		glm::vec3 finalColor = (1.0f - reflect_coef - refract_coef) * directLight + reflect_coef * reflectionColor + refract_coef * refractionColor;
//		finalColor += indirectLight;
//		finalColor += causticLight;
//		return min(finalColor, glm::vec3(1));
//	}
//
//	else {
//		return glm::vec3(0.0f); // Фон (например, черный цвет)
//	}
//
//
//}
//



const int WIDTH = 800;
const int HEIGHT = 600;


object create_cube(const glm::vec3& center, float side_length, const material& mat, float rotation_degrees = 0.0f) {
	float half_length = side_length / 2.0f;

	std::vector<glm::vec3> base_vertices = {
		glm::vec3(-half_length, -half_length, -half_length),
		glm::vec3(-half_length, -half_length,  half_length),
		glm::vec3(half_length, -half_length,  half_length),
		glm::vec3(half_length, -half_length, -half_length),
		glm::vec3(-half_length,  half_length, -half_length),
		glm::vec3(-half_length,  half_length,  half_length),
		glm::vec3(half_length,  half_length,  half_length),
		glm::vec3(half_length,  half_length, -half_length)
	};

	glm::mat4 rotation_matrix = glm::rotate(glm::mat4(1.0f),
		glm::radians(rotation_degrees),
		glm::vec3(0.0f, 1.0f, 0.0f));

	std::vector<glm::vec3> rotated_vertices;
	for (const auto& vertex : base_vertices) {
		glm::vec4 rotated = rotation_matrix * glm::vec4(vertex, 1.0f);
		rotated_vertices.push_back(center + glm::vec3(rotated));
	}

	object cube;
	cube.vertices = rotated_vertices;
	cube.faces = {
		face({0, 3, 2}), face({2, 1, 0}),
		face({7, 4, 5}), face({5, 6, 7}), 
		face({1, 2, 6}), face({6, 5, 1}), 
		face({0, 1, 5}), face({5, 4, 0}), 
		face({6, 2, 3}), face({3, 7, 6}), 
		face({7, 3, 0}), face({0, 4, 7})  
	};
	cube.material = mat;

	return cube;
}


void drawPhotons(sf::Image& image, const std::vector<photon>& nearestPhotons, const glm::vec3& cameraPosition, int width, int height) {
	
	//sf::Color photonColor(255, 255, 255);

	// Проходим по всем фотонам
	for (const photon& photon : nearestPhotons) {
		glm::vec3 direction = glm::normalize(photon.origin - cameraPosition);

		float u = (direction.x / direction.z + 1.0f) * 0.5f * width;
		float v = (1.0f - (direction.y / direction.z + 1.0f) * 0.5f) * height;

		int x = static_cast<int>(u);
		int y = static_cast<int>(v);

		int h = 5000;
		sf::Color pixelColor(
			static_cast<sf::Uint8>(min(photon.energy.r * h, 1.0f) * 255),
			static_cast<sf::Uint8>(min(photon.energy.g * h, 1.0f) * 255),
			static_cast<sf::Uint8>(min(photon.energy.b * h, 1.0f) * 255)
		);

		if (x >= 0 && x < width && y >= 0 && y < height) {
			image.setPixel(x, y, pixelColor);
		}
	}
}

void draw_near_photon(sf::Image& image, kd_node* tree, glm::vec3 target, glm::vec3 normal, const glm::vec3& cameraPosition, int width, int height, int k) {
	
	glm::vec3 plane_normal = normal;
	glm::vec3 plane_point = target;
	std::vector<photon_dist> nearest_photons = find_k_nearest_photons(tree, plane_point, k);

	std::vector<photon_dist> res;
	for (const auto& photon : nearest_photons) {
		float distance_to_plane = glm::dot(photon.p.origin - plane_point, plane_normal);
		if (std::abs(distance_to_plane) < 0.001f) { 
			res.push_back(photon);
		}
	}
	vector<photon> ph;
	for (auto& elem : res)
		ph.push_back(elem.p);

	drawPhotons(image, ph, cameraPosition, width, height);

	glm::vec3 direction = glm::normalize(target - cameraPosition);
	float u = (direction.x / direction.z + 1.0f) * 0.5f * width;
	float v = (1.0f - (direction.y / direction.z + 1.0f) * 0.5f) * height;

	int x = static_cast<int>(u);
	int y = static_cast<int>(v);
	sf::Color targetColor(0, 0, 255);
	int radius = 3;
	for (int dx = -radius; dx <= radius; ++dx) {
		for (int dy = -radius; dy <= radius; ++dy) {
			int px = x + dx;
			int py = y + dy;

			if (px >= 0 && px < width && py >= 0 && py < height) {
				image.setPixel(px, py, targetColor);
			}
		}
	}
}

void main() {
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray Tracing");
	window.setFramerateLimit(60);

	sf::Image image;
	image.create(WIDTH, HEIGHT, sf::Color::Black);


	glm::vec3 cube_p1 = glm::vec3(-1, -1, -1);
	glm::vec3 cube_p2 = glm::vec3(-1, -1, 1);
	glm::vec3 cube_p3 = glm::vec3(1, -1, 1);
	glm::vec3 cube_p4 = glm::vec3(1, -1, -1);
	glm::vec3 cube_p5 = glm::vec3(-1, 1, -1);
	glm::vec3 cube_p6 = glm::vec3(-1, 1, 1);
	glm::vec3 cube_p7 = glm::vec3(1, 1, 1);
	glm::vec3 cube_p8 = glm::vec3(1, 1, -1);


	object* down = new object(std::vector<glm::vec3> {cube_p1, cube_p2, cube_p3, cube_p4}, vector<face> {face({ 0, 1, 2 }), face({ 2, 3, 0 })}, material(glm::vec3(1.0f, 1.0f, 1.0f))); // down
	object* top = new object(std::vector<glm::vec3> {cube_p8, cube_p7, cube_p6, cube_p5}, vector<face> {face({ 0,1,2 }), face({ 2,3,0 })}, material(glm::vec3(0.9f, 0.9f, 1.0f))); // top
	object* back = new object(std::vector<glm::vec3> {cube_p2, cube_p6, cube_p7, cube_p3}, vector<face> {face({ 0,1,2 }), face({ 2,3,0 })}, material(glm::vec3(1.0f, 1.0f, 1.0f))); // back
	object* left = new object(std::vector<glm::vec3> {cube_p1, cube_p5, cube_p6, cube_p2}, vector<face> {face({ 0,1,2 }), face({ 2,3,0 })}, material(glm::vec3(0.6f, 0.6f, 1.0f))); // left
	object* right = new object(std::vector<glm::vec3> {cube_p4, cube_p3, cube_p7, cube_p8}, vector<face> {face({ 0,1,2 }), face({ 2,3,0 })}, material(glm::vec3(1.0f, 0.6f, 0.6f))); // right
	//object* front = new object(std::vector<glm::vec3> {cube_p1, cube_p4, cube_p8, cube_p5}, vector<face> {face({ 0,1,2 }), face({ 2,3,0 })}, material(glm::vec3(1.0f, 1.0f, 1.0f))); // front


	//auto light1 = std::make_shared<point_light> (glm::vec3(-0.0f, 0.99f, 0.2f), glm::vec3(1.0f, 1.0f, 1.0f), 0.8f);
	auto light2 = std::make_shared<point_light> (glm::vec3(0.99f, 0.0f, 0.2f), glm::vec3(1.0f, 1.0f, 1.0f), 0.8f);
	//auto light2 = std::make_shared<square_light>(
	//	glm::vec3(-0.99f, 0.0f, 0.5f), // Позиция (центр квадрата)
	//	glm::vec3(1.0f, 0.0f, 0.0f), // Направление (нормаль, светит вниз)
	//	0.5f,                        // Размер стороны квадрата
	//	0.5f,
	//	glm::vec3(1.0f, 0.8f, 0.7f), // Цвет света (белый)
	//	10.0f                         // Интенсивность
	//	);


	auto light1 = std::make_shared<square_light>(
		glm::vec3(-0.0f, 0.99f, 0.2f),
		glm::vec3(0.0f, -1.0f, 0.0f), 
		0.5f,                        
		0.2f,
		glm::vec3(1.0f, 1.0f, 1.0f), 
		20.0f                         
		);

	//auto light2 = std::make_shared<square_light>(
	//	glm::vec3(-0.99f, 0.5f, 0.2f), // Позиция (центр квадрата)
	//	glm::vec3(1.0f, 0.0f, 0.0f), // Направление (нормаль, светит вниз)
	//	0.3f,                        // Размер стороны квадрата
	//	0.3f,
	//	glm::vec3(1.0f, 1.0f, 1.0f), // Цвет света (белый)
	//	50.0f                         // Интенсивность
	//	);


	camera cam = camera(glm::vec3(0.0f, -0.0f, -1.0f));
	scene scene1;

	//object cub1 = create_cube(glm::vec3(-0.4, -0.75, 0.6), 0.5, material(glm::vec3(1.0f, 1.0f, 0.5f)), 50);
	////cub1.material.reflect_coef = 1.0;
	//object cub2 = create_cube(glm::vec3(0.45, -0.7, 0.55), 0.6, material(glm::vec3(1.0f, 1.0f, 1.0f)), -50);
	////cub2.material.reflect_coef = 0.01;
	//sphere* sphere1 = new sphere(glm::vec3(-0.4, -0.65, 0.5), 0.3, material(glm::vec3(1.0f, 1.0f, 1.0f)));
	//sphere1->material.reflect_coef = 1.0;
	////sphere1->material.refract_coef = 1.0;
	////sphere1->material.refract_ind = 1.6;

	//sphere* sphere2 = new sphere(glm::vec3(0.55, -0.65, 0.25), 0.25, material(glm::vec3(0.6f, 1.0f, 0.6f)));
	//sphere2->material.refract_coef = 1.0;
	//sphere2->material.refract_ind = 1.52;
	////sphere2->material.reflect_coef = 0.01;

	//sphere* sphere3 = new sphere(glm::vec3(0.5, 0.4, 0.5), 0.3, material(glm::vec3(1.0f, 1.0f, 1.0f)));
	//sphere3->material.reflect_coef = 1.0;
	////cub2->material.refract_coef = 1.0;
	////cub2->material.refract_ind = 1.5;
	////cub1.material.reflect_coef = 1.0f;
	////down.material.reflect_coef = 0.9;
	////right->material.reflect_coef = 0.9;
	////left->material.reflect_coef = 1.0;
	////front.material.reflect_coef = 1.0f;
	////back.material.reflect_coef = 1.0f;
	//scene1.objects = std::vector<object*>{ down, top, back, left, right
	//	//, front
	//	//,&cub1
	//	//,&cub2
	//	//,sphere1
	//	//,sphere2
	//	,sphere3
	//};

	vector<object> objects;
	ParseObj("water.obj", objects);


	objects[0].material = material(glm::vec3(1.0f, 1.0f, 1.0f));
	objects[1].material = material(glm::vec3(1.0f, 1.0f, 1.0f));
	objects[2].material = material(glm::vec3(1.0f, 1.0f, 1.0f));
	objects[3].material = material(glm::vec3(1.0f, 0.7f, 0.7f));
	objects[4].material = material(glm::vec3(0.7f, 0.7f, 1.0f));


	objects[5].material = material(glm::vec3(1.0f, 1.0f, 1.0f));
	objects[6].material = material(glm::vec3(1.0f, 1.0f, 1.0f));

	objects[5].material.refract_coef = 1.0;
	objects[5].material.refract_ind = 1.33;
	//objects[6].material.refract_coef = 1.0;
	//objects[6].material.refract_ind = 1.52;
	//objects[5].material.reflect_coef = 1.0;

	//objects[6].material.refract_coef = 1.0;
	//objects[6].material.refract_ind = 1.5;


	vector<object*> objs;
	for (auto obj : objects) {
		objs.push_back(new object(obj));
	}

	scene1.objects = objs;


	scene1.camera = cam;
	scene1.lights = std::vector<std::shared_ptr<light>>{ 
		light1
		//,light2 
	};

	
	trace_photons(scene1, 200000, 10);


	kd_node* tree=build_kd_tree(photons);
	kd_node* caustic_tree=build_kd_tree(caustic);
	kd_node* direct_tree=nullptr;
	//kd_node* direct_tree=build_kd_tree(direct_photons);



	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			float u = (2.0f * (x + 0.5f) / WIDTH - 1.0f);
			float v = 1.0f - (2.0f * (y + 0.5f) / HEIGHT);

			glm::vec3 direction = glm::normalize(glm::vec3(u, v, 1.0f));
			ray r = ray(scene1.camera.position, direction);
			glm::vec3 color = trace_ray(tree, direct_tree, caustic_tree, scene1, r, photons);

			sf::Color pixelColor(
				static_cast<sf::Uint8>(color.r * 255),
				static_cast<sf::Uint8>(color.g * 255),
				static_cast<sf::Uint8>(color.b * 255)
			);
			

			image.setPixel(x, y, pixelColor);
		}
	}


	//drawPhotons(image, caustic, scene1.camera.position, WIDTH, HEIGHT);
	//drawPhotons(image, photons, scene1.camera.position, WIDTH, HEIGHT);
	//drawPhotons(image, direct_photons, scene1.camera.position, WIDTH, HEIGHT);
	//draw_near_photon(image, tree, glm::vec3(-0.6, -1.0, 0.3), glm::vec3(0.0f,1.0f, 0.0f), scene1.camera.position, WIDTH, HEIGHT, 500);


	sf::Texture texture;
	texture.loadFromImage(image);

	sf::Sprite sprite;
	sprite.setTexture(texture);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}
		window.clear();
		window.draw(sprite);
		window.display();
	}

}

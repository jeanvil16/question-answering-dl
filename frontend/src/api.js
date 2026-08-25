const BASE = import.meta.env.VITE_API_URL || '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  const json = await res.json();
  if (!res.ok) {
    throw new Error(json?.error?.message || `Server error (${res.status})`);
  }
  return json;
}

export async function apiHealth() {
  return request('/health');
}

export async function apiPredict(context, question) {
  return request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context, question }),
  });
}

export async function apiModelInfo() {
  return request('/model/info');
}

export async function apiHistory() {
  return request('/model/history');
}

export const SAMPLES = [
  {
    title: 'Photosynthesis',
    context:
      'Photosynthesis is the process used by plants, algae and some bacteria to convert light energy into chemical energy. It takes place inside organelles called chloroplasts, which contain a green pigment called chlorophyll. During photosynthesis, plants absorb carbon dioxide from the air and water from the soil, and release oxygen as a by-product. The glucose produced is used for growth and energy storage. Photosynthesis is essential for life on Earth because it provides food and oxygen for nearly every living organism.',
    question: 'Where does photosynthesis take place?',
  },
  {
    title: 'Taj Mahal',
    context:
      'The Taj Mahal is a white marble mausoleum located in Agra, India, on the southern bank of the Yamuna River. It was commissioned in 1632 by the Mughal emperor Shah Jahan in memory of his favourite wife, Mumtaz Mahal, who died giving birth to their fourteenth child. Construction employed about twenty thousand artisans and was largely completed in 1653. Widely admired as the jewel of Muslim art in India, the Taj Mahal was designated a UNESCO World Heritage Site in 1983 and attracts millions of visitors every year.',
    question: 'Who commissioned the Taj Mahal?',
  },
  {
    title: 'Machine Learning',
    context:
      'Machine learning is a branch of artificial intelligence in which computers learn patterns from data instead of being programmed with explicit rules. In supervised learning a model is trained on labelled examples, while unsupervised learning finds hidden structure in unlabelled data. Reinforcement learning trains an agent through rewards and penalties. Deep learning uses artificial neural networks with many layers, known as deep neural networks, to learn rich representations. A model improves during training by adjusting its parameters to minimise a loss function, and its skill is measured on unseen test data.',
    question: 'What does deep learning use to learn rich representations?',
  },
  {
    title: 'Solar System',
    context:
      'The solar system consists of the Sun and everything that orbits around it, including eight planets. Jupiter is the largest planet, while Mercury is the smallest. Venus is the hottest planet because its thick atmosphere traps heat, whereas Neptune is the coldest and lies farthest from the Sun. Earth is the only planet known to support life. Saturn is famous for its spectacular ring system, and Mars is often called the red planet because iron minerals in its soil rust, giving the surface a reddish colour.',
    question: 'Which planet is called the red planet?',
  },
];
